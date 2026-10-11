#!/usr/bin/env python3
"""Run one WALL-E / EVE mission.

    python run.py configs/sample1.json --out outputs/sample1
"""
import argparse

from walle_eve import run_mission
from walle_eve.config import ConfigError


def main() -> int:
    ap = argparse.ArgumentParser(description="WALL-E and EVE's Signal: run a mission from a config file.")
    ap.add_argument("config", help="path to a mission config (.json / .yaml)")
    ap.add_argument("--out", default=None, help="directory for trace, report, maps and plot")
    ap.add_argument("--seed", type=int, default=None, help="override the environment's random seed")
    ap.add_argument("--agent-seed", type=int, default=0, help="seed for the agent's own sampling")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG summary")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the full trace while running")
    args = ap.parse_args()
    try:
        res = run_mission(args.config, out_dir=args.out, env_seed=args.seed, agent_seed=args.agent_seed,
                          make_plots=not args.no_plot, verbose=args.verbose)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2
    r = res["report"]
    ev = r.get("evaluation_vs_ground_truth", {})
    print(f"Mission '{r['mission']}': {r['outcome']}")
    print(f"  inferred location : {tuple(r['inferred_location'])}  (confidence {r['confidence']:.0%})")
    print(f"  true location     : {tuple(ev.get('true_eve_position', ()))}  (error {ev.get('position_error_cells')} cells)")
    rs = r["resources"]
    print(f"  resources         : {rs['actions']} actions, {rs['scans']} scans, {rs['recharges']} recharges, "
          f"battery spent {rs['battery_spent']:g}, time {rs['time_used']}/{rs['time_budget']}")
    print(f"  next step         : {r['recommended_next_step']}")
    if args.out:
        print(f"  files written to  : {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
