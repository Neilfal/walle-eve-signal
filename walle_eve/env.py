"""The simulated world: map, hidden EVE, hidden clues, noisy sensors and a resource budget.

The agent never touches this object's private state. It receives a *briefing* (map, costs,
sensor calibration, initial signal) and then learns about the world only through the
observations that ``step()`` returns.

Actions (dicts)
---------------
``{"type": "move", "dir": "N"|"S"|"E"|"W"}``  walk one cell
``{"type": "scan"}``                           noisy signal reading + reveals nearby clues
``{"type": "recharge"}``                       refill battery (only on a charger cell)
``{"type": "stop"}``                           end the mission and file the report
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .config import build_walls, normalize_config
from .geometry import DIRS, bearing_deg, chebyshev, compass_word, euclid


class MissionEnv:
    MAX_INVALID = 25  # safety net against agents that keep sending impossible actions

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self.cfg = normalize_config(config)
        cfg = self.cfg
        self.name: str = cfg["name"]
        self.width: int = cfg["grid"]["width"]
        self.height: int = cfg["grid"]["height"]
        self.walls = build_walls(cfg)
        self.start = tuple(cfg["start"])
        self.chargers = [tuple(c) for c in cfg["chargers"]]
        self.sensor: Dict[str, Any] = dict(cfg["sensor"])
        self.budget: Dict[str, Any] = cfg["budget"]
        self.costs: Dict[str, Dict[str, int]] = self.budget["costs"]
        self._eve = tuple(cfg["eve"]["position"])
        self._eve_status: str = cfg["eve"]["status"]
        self._clue_defs: List[Dict[str, Any]] = cfg["clues"]
        self._seed = cfg["seed"] if seed is None else seed
        self.reset()

    # ------------------------------------------------------------------ public API
    def briefing(self) -> Dict[str, Any]:
        """Everything the agent is allowed to know before the mission starts."""
        sensor = {k: v for k, v in self.sensor.items() if k != "spoof_prob"}
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "walls": [list(map(int, w)) for w in self.cfg["grid"]["walls"]],
            "start": list(self.start),
            "chargers": [list(c) for c in self.chargers],
            "budget": {
                "battery": self.budget["battery"],
                "time": self.budget["time"],
                "max_recharges": self.budget["max_recharges"],
                "costs": {k: dict(v) for k, v in self.costs.items()},
            },
            "sensor": sensor,
            "initial_signal": dict(self.cfg["initial_signal"]),
        }

    def reset(self) -> Dict[str, Any]:
        self.rng = np.random.default_rng(self._seed)
        self.pos = self.start
        self.battery: float = self.budget["battery"]
        self.time_used = 0
        self.steps = 0
        self.recharges_used = 0
        self.invalid_count = 0
        self.spoofed_scans = 0
        self.done = False
        self.done_reason: Optional[str] = None
        # Evidence is generated up-front, in config order, so it never depends on what the
        # agent does (same seed -> same clue text regardless of discovery order).
        self._evidence = {c["id"]: self._make_evidence(c) for c in self._clue_defs}
        self._discovered = set()
        return self._obs("start", "Mission start.")

    @property
    def time_left(self) -> int:
        return self.budget["time"] - self.time_used

    @property
    def recharges_left(self) -> int:
        return self.budget["max_recharges"] - self.recharges_used

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("The mission is over; call reset() to start again.")
        self.steps += 1
        kind = action.get("type")
        if kind == "stop":
            self.done, self.done_reason = True, "agent_stop"
            return self._obs("stopped", "Agent ended the mission.")
        if kind not in ("move", "scan", "recharge"):
            return self._invalid(f"Unknown action {action!r}.")
        cost = self.costs[kind]
        if cost["time"] > self.time_left:
            self.done, self.done_reason = True, "time_up"
            return self._obs("time_up", "Not enough time left for that action.", ok=False)
        if cost["battery"] > self.battery:
            return self._invalid("Not enough battery for that action.", "insufficient_battery")
        if kind == "move":
            return self._do_move(action, cost)
        if kind == "scan":
            return self._do_scan(cost)
        return self._do_recharge(cost)

    def ground_truth(self) -> Dict[str, Any]:
        """Hidden facts. For evaluation and plots only; the agent never sees this."""
        return {
            "eve_position": list(self._eve),
            "eve_status": self._eve_status,
            "spoofed_scans": self.spoofed_scans,
            "clues": [
                {"id": c["id"], "type": c["type"], "position": list(c["position"]),
                 "decoy": bool(c["decoy"]), "discovered": c["id"] in self._discovered}
                for c in self._clue_defs
            ],
        }

    # ------------------------------------------------------------------ actions
    def _do_move(self, action, cost):
        d = action.get("dir")
        if d not in DIRS:
            return self._invalid(f"Bad direction {d!r}.")
        nx, ny = self.pos[0] + DIRS[d][0], self.pos[1] + DIRS[d][1]
        if not (0 <= nx < self.width and 0 <= ny < self.height) or self.walls[ny, nx]:
            return self._invalid(f"Blocked: ({nx},{ny}) is a wall or off the map.", "blocked")
        self._spend(cost)
        self.pos = (nx, ny)
        return self._after_action("moved", f"Moved {d} to ({nx},{ny}).")

    def _do_recharge(self, cost):
        if self.pos not in self.chargers:
            return self._invalid("Recharge needs a charging station.", "not_at_charger")
        if self.recharges_left <= 0:
            return self._invalid("No recharges left in the budget.", "no_recharges_left")
        self._spend(cost)
        self.battery = self.budget["battery"]
        self.recharges_used += 1
        return self._after_action("recharged", f"Battery refilled to {self.battery:g}.")

    def _do_scan(self, cost):
        self._spend(cost)
        s = self.sensor
        d = euclid(self.pos, self._eve)
        in_view = d <= s["visual_range"]
        spoofed = bool(self.rng.random() < s["spoof_prob"])
        if spoofed:
            self.spoofed_scans += 1
            strength = float(self.rng.uniform(0.0, 1.0))
            bearing = float(self.rng.uniform(0.0, 360.0))
        else:
            mean = s["s0"] * math.exp(-d / s["decay_length"])
            strength = max(0.0, float(mean + self.rng.normal(0.0, s["strength_sigma"])))
            sigma_b = s["bearing_sigma_base_deg"] + s["bearing_sigma_per_cell_deg"] * d
            bearing = float((bearing_deg(self.pos, self._eve) + self.rng.normal(0.0, sigma_b)) % 360.0)
        clues = []
        for c in self._clue_defs:
            if chebyshev(self.pos, tuple(c["position"])) <= s["clue_scan_radius"]:
                ev = dict(self._evidence[c["id"]])
                ev["already_known"] = c["id"] in self._discovered
                self._discovered.add(c["id"])
                clues.append(ev)
        scan = {
            "strength": round(strength, 4),
            "bearing_deg": round(bearing, 1),
            "eve_in_view": bool(in_view),
            "eve_position": list(self._eve) if in_view else None,
            "eve_status": self._eve_status if in_view else None,
            "clues": clues,
        }
        msg = f"Scanned at {self.pos}: strength {scan['strength']:.3f}, bearing {scan['bearing_deg']:.0f} deg."
        if in_view:
            msg += f" EVE IS IN VIEW at {self._eve}!"
        new = [c["id"] for c in clues if not c["already_known"]]
        if new:
            msg += f" New clue(s): {', '.join(new)}."
        return self._after_action("scanned", msg, scan=scan)

    # ------------------------------------------------------------------ internals
    def _make_evidence(self, c: Dict[str, Any]) -> Dict[str, Any]:
        pos = tuple(c["position"])
        base = {"id": c["id"], "type": c["type"], "position": list(pos)}
        if c["type"] == "bearing":
            target = tuple(c["decoy_target"]) if c["decoy"] else self._eve
            b = float((bearing_deg(pos, target) + self.rng.normal(0.0, c["sigma"])) % 360.0)
            base.update(bearing_deg=round(b, 1), sigma=c["sigma"],
                        text=f"Wheel tracks and a dragged cable head {compass_word(b)} (bearing ~{b:.0f} deg).")
        elif c["type"] == "range":
            true_d = euclid(pos, self._eve) + (c["decoy_offset"] if c["decoy"] else 0.0)
            d = max(0.0, float(true_d + self.rng.normal(0.0, c["sigma"])))
            base.update(distance=round(d, 2), sigma=c["sigma"],
                        text=f"Old beacon log: the signal source is roughly {d:.1f} cells from here.")
        else:
            base.update(hint=c["hint"],
                        text=f"Scorch marks and residue suggest EVE is {str(c['hint']).replace('_', ' ')}.")
        return base

    def _spend(self, cost):
        self.battery -= cost["battery"]
        self.time_used += cost["time"]

    def _after_action(self, event, message, scan=None):
        if self.time_left <= 0:
            self.done, self.done_reason = True, "time_up"
        else:
            cheapest = min(self.costs["move"]["battery"], self.costs["scan"]["battery"])
            can_refill = (self.pos in self.chargers and self.recharges_left > 0
                          and self.time_left >= self.costs["recharge"]["time"])
            if self.battery < cheapest and not can_refill:
                self.done, self.done_reason = True, "out_of_battery"
        return self._obs(event, message, scan=scan)

    def _invalid(self, message, event="invalid"):
        self.invalid_count += 1
        if self.invalid_count >= self.MAX_INVALID:
            self.done, self.done_reason = True, "too_many_invalid_actions"
        return self._obs(event, message, ok=False)

    def _obs(self, event, message, ok=True, scan=None) -> Dict[str, Any]:
        return {
            "step": self.steps,
            "event": event,
            "ok": ok,
            "message": message,
            "position": list(self.pos),
            "battery": self.battery,
            "time_used": self.time_used,
            "time_left": self.time_left,
            "recharges_left": self.recharges_left,
            "scan": scan,
            "done": self.done,
            "done_reason": self.done_reason,
        }
