"""Final mission report: evidence found, inferred location/status, confidence, next step."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .geometry import euclid

STATUS_ADVICE = {
    "low_power": "EVE reports low power: carry a spare power cell and prioritise docking.",
    "damaged": "EVE looks damaged: bring a repair kit.",
    "dormant": "EVE appears dormant: approach slowly and try a wake-up signal.",
    "active": "EVE appears active: expect it to respond to a greeting ping.",
}


def build_report(agent, env=None) -> Dict[str, Any]:
    belief = agent.belief
    confirmed = agent.confirmed is not None
    est = tuple(agent.confirmed) if confirmed else belief.estimate()
    conf = 1.0 if confirmed else belief.confidence()
    if confirmed:
        outcome = "EVE_LOCATED"
    elif conf >= agent.params["conf_stop"]:
        outcome = "EVE_LOCALIZED"
    elif conf >= 0.5:
        outcome = "EVE_PROBABLY_LOCATED"
    else:
        outcome = "INCONCLUSIVE"

    mb = agent.costs["move"]["battery"]
    steps = agent.path_len(est)
    home = agent.home_dist.get(agent.pos)
    status = agent.status_estimate()
    next_step = _recommend(outcome, est, conf, steps, mb, agent, status)

    evidence = []
    for item in agent.evidence_log:
        row = dict(item)
        if item["source"].startswith("clue "):
            row["consistent_with_final_belief"] = agent.clue_consistency(agent.clues[item["source"][5:]])
        evidence.append(row)

    spent_time = agent.briefing["budget"]["time"] - agent.time_left
    report: Dict[str, Any] = {
        "mission": agent.briefing["name"],
        "outcome": outcome,
        "inferred_location": list(est),
        "confidence": round(float(conf), 3),
        "confidence_definition": "probability that EVE is within one cell of the inferred location",
        "visually_confirmed": confirmed,
        "credible_region_cells_90pct": belief.credible_cells(0.9),
        "inferred_status": status,
        "top_hypotheses": [{"cell": list(c), "probability": round(p, 3)} for c, p in belief.top_cells(5)],
        "evidence": evidence,
        "clues_found": len(agent.clues),
        "strategy_timeline": agent.strategy_timeline,
        "termination": {"agent_reason": agent.stop_reason,
                        "environment_reason": env.done_reason if env else None},
        "resources": {
            "actions": agent.stats["actions"], "moves": agent.stats["moves"],
            "scans": agent.stats["scans"], "recharges": agent.stats["recharges"],
            "battery_spent": agent.stats["battery_spent"],
            "battery_left": agent.battery, "battery_capacity": agent.max_battery,
            "time_used": spent_time, "time_budget": agent.briefing["budget"]["time"],
            "final_position": list(agent.pos),
            "steps_to_nearest_charger": home,
            "stranded": bool(env and env.done_reason == "out_of_battery"),
        },
        "recommended_next_step": next_step,
    }
    if env is not None:
        truth = env.ground_truth()
        err = euclid(est, tuple(truth["eve_position"]))
        report["evaluation_vs_ground_truth"] = {
            "note": "Computed by the evaluator after the mission; the agent never saw this.",
            "true_eve_position": truth["eve_position"],
            "true_eve_status": truth["eve_status"],
            "position_error_cells": round(err, 2),
            "located_within_visual_range": bool(err <= agent.sensor["visual_range"]),
            "status_correct": (status == truth["eve_status"]) if status != "unknown" else None,
            "spoofed_scans_received": truth["spoofed_scans"],
            "decoy_clues_found": [c["id"] for c in truth["clues"] if c["decoy"] and c["discovered"]],
        }
    return report


def _recommend(outcome, est, conf, steps, mb, agent, status) -> str:
    cost = None if steps is None else steps * mb
    if outcome == "INCONCLUSIVE":
        tops = ", ".join(f"{tuple(c)} ({p:.0%})" for c, p in agent.belief.top_cells(3))
        return (f"Evidence is inconclusive ({conf:.0%} confidence). Recharge, then search the most likely cells first: "
                f"{tops}. Consider a larger battery/time budget or scanning from more directions.")
    reach = ""
    if cost is not None:
        if agent.battery >= cost + agent.costs["scan"]["battery"]:
            reach = f" It is {steps} moves away (~{cost:g} battery); the current charge ({agent.battery:g}) is enough."
        else:
            reach = (f" It is {steps} moves away (~{cost:g} battery) but only {agent.battery:g} remains: "
                     f"recharge first (nearest charger: {agent.home_dist.get(agent.pos)} steps away).")
    if outcome == "EVE_LOCATED":
        return f"Go to EVE at {est} and dock/meet.{reach} {STATUS_ADVICE.get(status, '')}".strip()
    return (f"Move to within one cell of {est} and scan once to visually confirm EVE (confidence {conf:.0%}).{reach} "
            f"{STATUS_ADVICE.get(status, '')}").strip()


def report_markdown(r: Dict[str, Any]) -> str:
    res = r["resources"]
    L = [f"# Mission report: {r['mission']}", ""]
    L += [f"**Outcome:** `{r['outcome']}`  ", f"**Inferred location of EVE:** {tuple(r['inferred_location'])}  ",
          f"**Confidence:** {r['confidence']:.0%} ({r['confidence_definition']})  ",
          f"**Visually confirmed:** {'yes' if r['visually_confirmed'] else 'no'}  ",
          f"**Inferred EVE status:** {r['inferred_status']}  ",
          f"**90% credible region:** {r['credible_region_cells_90pct']} cells", ""]
    L += ["## Recommended next step", "", r["recommended_next_step"], ""]
    L += ["## Evidence found", "", "| step | source | what it says | vs. final belief |", "|---|---|---|---|"]
    for e in r["evidence"]:
        L.append(f"| {e['step']} | {e['source']} @ {tuple(e['position'])} | {e['text']} | "
                 f"{e.get('consistent_with_final_belief', '-')} |")
    L += ["", "## Top hypotheses", "", "| cell | probability |", "|---|---|"]
    L += [f"| {tuple(h['cell'])} | {h['probability']:.1%} |" for h in r["top_hypotheses"]]
    L += ["", "## Strategy timeline", ""]
    for t in r["strategy_timeline"]:
        L.append(f"- after action {t['action_no']}: **{t['from']} -> {t['to']}**. {t['reason']}")
    L += ["", "## Resources", "",
          f"- actions: {res['actions']} ({res['moves']} moves, {res['scans']} scans, {res['recharges']} recharges)",
          f"- battery spent: {res['battery_spent']:g}; left {res['battery_left']:g}/{res['battery_capacity']}",
          f"- time used: {res['time_used']}/{res['time_budget']}",
          f"- final position: {tuple(res['final_position'])} ({res['steps_to_nearest_charger']} steps from a charger)",
          f"- stranded: {'yes' if res['stranded'] else 'no'}",
          f"- stopped because: {r['termination']['agent_reason'] or r['termination']['environment_reason']}", ""]
    ev = r.get("evaluation_vs_ground_truth")
    if ev:
        L += ["## Evaluation against ground truth (evaluator only)", "",
              f"- true EVE position: {tuple(ev['true_eve_position'])}; error {ev['position_error_cells']} cells; "
              f"within visual range: {ev['located_within_visual_range']}",
              f"- true status: {ev['true_eve_status']} (inferred correctly: "
              f"{'n/a, no status evidence found' if ev['status_correct'] is None else ev['status_correct']})",
              f"- spoofed scans received: {ev['spoofed_scans_received']}; "
              f"decoy clues found: {ev['decoy_clues_found'] or 'none'}", ""]
    return "\n".join(L)
