"""WALL-E: an autonomous search agent.

Decision loop (one ``act()`` call at a time)
--------------------------------------------
1. Stop if EVE has been seen, or the belief is concentrated enough.
2. Look after the battery: top up at a charger, or turn back when charge runs low.
3. Otherwise pick the next *scan point*: among all cells that are affordable (the agent must
   always keep enough charge to get back to a charger), choose the one with the best
   ``expected information gain - travel cost``. Walk there along a shortest path and scan.
4. Every observation updates the Bayesian belief; the next plan is made from the new belief.

Everything the agent knows comes from the *briefing* and from observations. It has no access
to the environment object.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .belief import (Belief, bearing_likelihood, bearing_sigma, expected_strength,
                     strength_likelihood)
from .geometry import Cell, bfs, chebyshev, direction_between, follow_parents

DEFAULT_PARAMS: Dict[str, Any] = {
    "conf_stop": 0.90,            # stop when P(EVE within 1 cell of estimate) reaches this
    "min_scans_before_stop": 2,   # ...but never on fewer scans than this
    "low_battery_frac": 0.30,     # below this fraction away from base: head back to recharge
    "topup_frac": 0.70,           # at base below this fraction: recharge before leaving
    "reserve_margin": 2,          # extra battery kept on top of the return trip
    "lam": 0.03,                  # bits of information one battery unit is "worth"
    "clue_bonus": 0.25,           # bits of bonus for scanning an area not yet searched for clues
    "mc_samples": 12,             # Monte-Carlo samples used to estimate information gain
    "max_scans_same_cell": 2,     # avoid scanning the same cell over and over
    "outlier_prob": 0.05,         # robustness: chance any signal reading is garbage
    "clue_outlier_prob": 0.20,    # robustness: chance any clue is misleading (decoy)
    "min_time_after_recharge": 8, # only recharge if this much time would remain afterwards
}


class Agent:
    def __init__(self, briefing: Dict[str, Any], seed: int = 0, params: Optional[Dict[str, Any]] = None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        b = briefing
        self.briefing = b
        self.width, self.height = b["width"], b["height"]
        self.walls = np.zeros((self.height, self.width), dtype=bool)
        for x, y in b["walls"]:
            self.walls[y, x] = True
        self.sensor = b["sensor"]
        self.costs = b["budget"]["costs"]
        self.max_battery = b["budget"]["battery"]
        self.chargers = [tuple(c) for c in b["chargers"]]
        self.rng = np.random.default_rng(seed)
        self.belief = Belief(self.walls, self.sensor, self.params["outlier_prob"],
                             self.params["clue_outlier_prob"])

        # what the agent knows about itself
        self.pos: Cell = tuple(b["start"])
        self.battery = float(self.max_battery)
        self.time_left = b["budget"]["time"]
        self.recharges_left = b["budget"]["max_recharges"]

        # distance-to-base field (multi-source BFS from every charger)
        self.home_dist, self.home_parent = bfs(self.chargers, self._is_free, self.width, self.height)

        # knowledge gathered during the mission
        self.queue: deque = deque()
        self.clues: Dict[str, Dict[str, Any]] = {}
        self.evidence_log: List[Dict[str, Any]] = []
        self.scan_log: List[Cell] = []
        self.scan_counts: Dict[Cell, int] = {}
        self.visited: List[Cell] = [self.pos]
        self.confirmed: Optional[Cell] = None
        self.eve_status: Optional[str] = None
        self.stop_reason: Optional[str] = None

        # bookkeeping for the report / visualisations
        self.mode = "start"
        self.strategy_timeline: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.frames: List[Dict[str, Any]] = []
        self.stats = {"moves": 0, "scans": 0, "recharges": 0, "battery_spent": 0.0,
                      "invalid": 0, "actions": 0}

    # ================================================================== helpers
    def _is_free(self, cell: Cell) -> bool:
        return not self.walls[cell[1], cell[0]]

    def path_len(self, target: Cell) -> Optional[int]:
        dist, _ = bfs([self.pos], self._is_free, self.width, self.height)
        return dist.get(tuple(target))

    # ================================================================== life-cycle
    def begin(self, obs: Dict[str, Any]) -> None:
        """Use the initial signal from the mission config as the first piece of evidence."""
        sig = self.briefing["initial_signal"]
        self.belief.update_strength(self.pos, sig["strength"], sigma=2 * self.sensor["strength_sigma"])
        self.belief.update_bearing(self.pos, sig["bearing_deg"], sigma=sig["bearing_sigma_deg"])
        self.evidence_log.append({
            "step": 0, "source": "initial_signal", "position": list(self.pos), "kind": "signal",
            "text": (f"Initial signal (t={sig['timestamp']}): strength {sig['strength']:.2f}, "
                     f"bearing {sig['bearing_deg']:.0f} deg (+/-{sig['bearing_sigma_deg']:.0f})."),
        })
        self._snapshot(0, "start")

    def observe(self, action: Dict[str, Any], obs: Dict[str, Any]) -> None:
        """Update position/resources and fold the observation into the belief."""
        self.pos = tuple(obs["position"])
        self.battery = obs["battery"]
        self.time_left = obs["time_left"]
        self.recharges_left = obs["recharges_left"]
        if not obs["ok"]:
            self.stats["invalid"] += 1
            self.queue.clear()          # the plan no longer matches reality: re-plan
            return
        kind = action["type"]
        self.stats["actions"] += 1
        if kind == "move":
            self.stats["moves"] += 1
            self.stats["battery_spent"] += self.costs["move"]["battery"]
            self.visited.append(self.pos)
        elif kind == "scan":
            self.stats["scans"] += 1
            self.stats["battery_spent"] += self.costs["scan"]["battery"]
            self._ingest_scan(obs["scan"], obs["step"])
        elif kind == "recharge":
            self.stats["recharges"] += 1
        self._snapshot(obs["step"], self.mode, scan=(kind == "scan"))

    # ================================================================== evidence
    def _ingest_scan(self, scan: Dict[str, Any], step: int) -> None:
        pos = self.pos
        self.scan_log.append(pos)
        self.scan_counts[pos] = self.scan_counts.get(pos, 0) + 1
        if scan["eve_in_view"]:
            self.confirmed = tuple(scan["eve_position"])
            self.eve_status = scan["eve_status"]
            self.belief.update_visual(pos, True, self.confirmed)
            self.evidence_log.append({
                "step": step, "source": "scan", "position": list(pos), "kind": "visual",
                "text": f"EVE seen directly at {self.confirmed}; status: {self.eve_status}."})
        else:
            self.belief.update_visual(pos, False)
            self.belief.update_strength(pos, scan["strength"])
            self.belief.update_bearing(pos, scan["bearing_deg"])
            self.evidence_log.append({
                "step": step, "source": "scan", "position": list(pos), "kind": "signal",
                "text": f"Signal at {pos}: strength {scan['strength']:.3f}, bearing {scan['bearing_deg']:.0f} deg; "
                        f"EVE not within visual range."})
        for clue in scan["clues"]:
            if clue["id"] in self.clues:
                continue
            self.clues[clue["id"]] = dict(clue, step=step)
            cpos = tuple(clue["position"])
            if clue["type"] == "bearing":
                self.belief.update_clue_bearing(cpos, clue["bearing_deg"], clue["sigma"])
            elif clue["type"] == "range":
                self.belief.update_clue_range(cpos, clue["distance"], clue["sigma"])
            self.evidence_log.append({
                "step": step, "source": f"clue {clue['id']}", "position": list(cpos),
                "kind": clue["type"], "text": clue["text"]})
            if clue["type"] == "status" and self.eve_status is None:
                pass  # status hints are summarised in the report

    def status_estimate(self) -> str:
        if self.eve_status:
            return self.eve_status
        hints = [c["hint"] for c in self.clues.values() if c["type"] == "status"]
        return max(set(hints), key=hints.count) if hints else "unknown"

    def clue_consistency(self, clue: Dict[str, Any]) -> str:
        """Compare a clue with the final belief: 'consistent', 'conflicting' (beyond 2.5 sigma) or 'n/a'."""
        est = self.belief.estimate()
        cpos = tuple(clue["position"])
        if clue["type"] == "bearing":
            _, bear = self.belief.geo.dist_bearing(cpos)
            diff = abs(((clue["bearing_deg"] - bear[est[1], est[0]] + 180) % 360) - 180)
            return "consistent" if diff <= 2.5 * clue["sigma"] else "conflicting"
        if clue["type"] == "range":
            dist, _ = self.belief.geo.dist_bearing(cpos)
            return "consistent" if abs(clue["distance"] - dist[est[1], est[0]]) <= 2.5 * clue["sigma"] else "conflicting"
        return "n/a"

    # ================================================================== decision making
    def act(self) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Return ``(action, reasoning, meta)`` for the next step."""
        if self.queue:
            action, reason, mode, meta = self.queue.popleft()
            return action, reason, dict(meta, mode=mode)
        action, reason, mode, meta = self._plan()
        self._set_mode(mode, reason)
        return action, reason, dict(meta, mode=mode)

    def _set_mode(self, mode: str, reason: str) -> None:
        if mode != self.mode:
            self.strategy_timeline.append({"action_no": self.stats["actions"], "from": self.mode,
                                           "to": mode, "reason": reason})
            self.mode = mode

    def _stop(self, reason: str):
        self.stop_reason = reason
        return {"type": "stop"}, reason, "finish", {}

    def _plan(self):
        P = self.params
        belief = self.belief
        est, conf, ent = belief.estimate(), belief.confidence(), belief.entropy()
        mb, sb = self.costs["move"]["battery"], self.costs["scan"]["battery"]
        mt, rt = self.costs["move"]["time"], self.costs["recharge"]["time"]
        frac = self.battery / self.max_battery
        at_home = self.pos in self.chargers
        home_steps = self.home_dist[self.pos]

        # ---- 1. finished? ------------------------------------------------------------
        if self.confirmed is not None:
            return self._stop(f"EVE visually confirmed at {self.confirmed}; mission goal reached.")
        if conf >= P["conf_stop"] and len(self.scan_log) >= P["min_scans_before_stop"]:
            return self._stop(f"Belief is concentrated: {conf:.0%} probability that EVE is within one cell of {est}.")

        # ---- 2. battery management ---------------------------------------------------
        recharge_ok = (self.recharges_left > 0 and self.time_left >= rt + P["min_time_after_recharge"])
        if at_home and recharge_ok and frac < P["topup_frac"]:
            return ({"type": "recharge"},
                    f"At a charger with {self.battery:g}/{self.max_battery} battery ({frac:.0%}): topping up before the next excursion.",
                    "recharge", {})
        if (not at_home and recharge_ok and frac <= P["low_battery_frac"]
                and self.time_left >= home_steps * mt + rt + P["min_time_after_recharge"]):
            reason = (f"Battery low ({self.battery:g}/{self.max_battery}, {frac:.0%}). Turning back: "
                      f"{home_steps} steps to the nearest charger.")
            return self._go_home(reason)

        # ---- 3. choose the next scan point ------------------------------------------
        cands, parent = self._evaluate_candidates()
        if not cands:
            if at_home:
                if recharge_ok and frac < 0.95:
                    return ({"type": "recharge"},
                            "No scan point is affordable on the current charge; recharging.", "recharge", {})
                return self._stop("No affordable action is left (battery, time or recharges exhausted).")
            return self._go_home("No scan point is affordable and still leaves a safe return; turning back.")

        cands.sort(key=lambda c: c["utility"], reverse=True)
        best = cands[0]
        if best["utility"] < 0 and conf >= 0.5:
            return self._stop(f"Best remaining action (utility {best['utility']:.2f}) is not worth its cost; "
                              f"stopping with {conf:.0%} confidence in {est}.")
        path = follow_parents(parent, best["cell"])[::-1]          # [pos, ..., target]
        near_estimate = (np.hypot(best["cell"][0] - est[0], best["cell"][1] - est[1])
                         <= self.sensor["visual_range"] + 1.0)
        mode = "verify" if (near_estimate and conf >= 0.4) else "explore"
        why = (f"Belief peak {est} ({conf:.0%} within 1 cell), entropy {ent:.2f} bits. "
               f"Battery {self.battery:g}/{self.max_battery}. Next scan point {best['cell']}: "
               f"{best['steps']} steps, expected information gain {best['gain']:.2f} bits"
               f"{' + new-area bonus' if best['bonus'] else ''}, cost {best['cost']:g} battery; "
               f"{best['battery_after']:g} would remain vs {best['reserve']:g} needed to get back to a charger.")
        if len(cands) > 1:
            alt = cands[1]
            why += f" Runner-up {alt['cell']} scored {alt['utility']:.2f} vs {best['utility']:.2f}."
        meta = {"target": list(best["cell"]), "utility": round(best["utility"], 3),
                "expected_gain_bits": round(best["gain"], 3)}
        for a, b in zip(path[:-1], path[1:]):
            self.queue.append(({"type": "move", "dir": direction_between(a, b)},
                               f"En route to scan point {best['cell']}.", mode, meta))
        self.queue.append(({"type": "scan"}, f"Arrived at {best['cell']}: taking a reading.", mode, meta))
        action = self.queue.popleft()[0]       # first step of the plan; the rest stay queued
        return action, why, mode, meta

    def _go_home(self, reason: str):
        path = follow_parents(self.home_parent, self.pos)           # [pos, ..., charger]
        if len(path) < 2:
            return self._stop("Out of options at base.")
        for a, b in zip(path[:-1], path[1:]):
            self.queue.append(({"type": "move", "dir": direction_between(a, b)},
                               "Returning to the charger.", "return", {"target": list(path[-1])}))
        action, _, _, meta = self.queue.popleft()
        return action, reason, "return", meta

    # ------------------------------------------------------------------ candidate scoring
    def _evaluate_candidates(self):
        P = self.params
        mb, sb = self.costs["move"]["battery"], self.costs["scan"]["battery"]
        mt, st = self.costs["move"]["time"], self.costs["scan"]["time"]
        dist, parent = bfs([self.pos], self._is_free, self.width, self.height)
        feasible = []
        for cell, steps in dist.items():
            if self.scan_counts.get(cell, 0) >= P["max_scans_same_cell"]:
                continue
            cost = steps * mb + sb
            reserve = self.home_dist[cell] * mb + P["reserve_margin"]
            if self.battery - cost < reserve:        # must always be able to get back to a charger
                continue
            if steps * mt + st > self.time_left:
                continue
            feasible.append((cell, steps, cost, reserve))
        if not feasible:
            return [], parent
        gains = self._info_gain([f[0] for f in feasible])
        radius = self.sensor["clue_scan_radius"]
        out = []
        for (cell, steps, cost, reserve), gain in zip(feasible, gains):
            novel = all(chebyshev(cell, s) > radius for s in self.scan_log)
            bonus = P["clue_bonus"] if novel else 0.0
            out.append({"cell": cell, "steps": steps, "cost": cost, "reserve": reserve,
                        "battery_after": self.battery - cost, "gain": gain, "bonus": bonus,
                        "utility": gain + bonus - P["lam"] * cost})
        return out, parent

    def _info_gain(self, cells: List[Cell]) -> List[float]:
        """Monte-Carlo estimate of the entropy reduction (bits) from scanning at each cell.

        Draw K hypotheses for EVE's position from the current belief, simulate the scan the
        agent would get under each (noisy strength + bearing, or a direct sighting), compute the
        posterior entropy, and average. Common random numbers are reused across candidate cells
        so the comparison between cells is low-noise.
        """
        P, sensor, belief = self.params, self.sensor, self.belief
        K = P["mc_samples"]
        ys, xs = belief.sample(self.rng, K)
        z_s, z_b = self.rng.standard_normal(K), self.rng.standard_normal(K)
        h0 = belief.entropy()
        vr, eps = sensor["visual_range"], P["outlier_prob"]
        gains = []
        for c in cells:
            D, B = belief.geo.dist_bearing(c)
            dh, bh = D[ys, xs], B[ys, xs]
            seen = dh <= vr
            s_obs = expected_strength(sensor, dh) + sensor["strength_sigma"] * z_s
            b_obs = bh + bearing_sigma(sensor, dh) * z_b
            lik = (strength_likelihood(sensor, s_obs[:, None, None], D[None], eps)
                   * bearing_likelihood(b_obs[:, None, None], B[None], bearing_sigma(sensor, D)[None], eps, D[None])
                   * np.where(D <= vr, 0.01, 1.0)[None])
            post = belief.p[None] * lik
            post = post / post.sum(axis=(1, 2), keepdims=True)
            ent = -np.where(post > 0, post * np.log2(np.where(post > 0, post, 1.0)), 0.0).sum(axis=(1, 2))
            ent[seen] = 0.0                              # a sighting leaves no uncertainty
            gains.append(max(0.0, float(h0 - ent.mean())))
        return gains

    # ================================================================== logging helpers
    def _snapshot(self, step: int, mode: str, scan: bool = False) -> None:
        est = self.confirmed or self.belief.estimate()
        self.history.append({"step": step, "battery": self.battery, "time_left": self.time_left,
                             "position": list(self.pos), "entropy": self.belief.entropy(),
                             "confidence": 1.0 if self.confirmed else self.belief.confidence(),
                             "estimate": list(est), "mode": mode})
        if scan or step == 0:
            self.frames.append({"step": step, "p": self.belief.p.copy(), "position": self.pos,
                                "path": list(self.visited), "estimate": est,
                                "clues": {k: v["position"] for k, v in self.clues.items()}})
