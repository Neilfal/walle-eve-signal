#!/usr/bin/env python3
"""Generalisation benchmark: run the same agent on many random missions.

    python benchmark.py --n 20 --seed 100
"""
import argparse
import csv
import os
import statistics as st
import time

from walle_eve import run_mission
from walle_eve.generator import random_config

PROFILES = {
    # name: kwargs for random_config
    "standard": dict(n_clues=4, n_decoys=0, spoof_prob=0.0, noise=1.0, battery=40),
    "hard": dict(width=20, height=20, wall_density=0.15, n_clues=5, n_decoys=2, spoof_prob=0.10,
                 noise=1.5, battery=45, second_charger=True, time_budget=220),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="missions per profile")
    ap.add_argument("--seed", type=int, default=100, help="first mission seed")
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, summary = [], []
    for name, kw in PROFILES.items():
        t0 = time.time()
        res = []
        for seed in range(args.seed, args.seed + args.n):
            r = run_mission(random_config(seed, **kw), make_plots=False, agent_seed=seed)["report"]
            ev, rs = r["evaluation_vs_ground_truth"], r["resources"]
            res.append(dict(profile=name, seed=seed, outcome=r["outcome"], confidence=r["confidence"],
                            error=ev["position_error_cells"], success=int(ev["located_within_visual_range"]),
                            confirmed=int(r["visually_confirmed"]), scans=rs["scans"], recharges=rs["recharges"],
                            battery_spent=rs["battery_spent"], time_used=rs["time_used"],
                            stranded=int(rs["stranded"]), status_ok=int(ev["status_correct"] is True),
                            spoofed=ev["spoofed_scans_received"], decoys_found=len(ev["decoy_clues_found"])))
        rows += res
        ok = [x for x in res if x["success"]]
        summary.append(dict(
            profile=name, missions=len(res), success=f"{100 * len(ok) / len(res):.0f}%",
            confirmed=f"{100 * sum(x['confirmed'] for x in res) / len(res):.0f}%",
            median_error=f"{st.median(x['error'] for x in res):.2f}",
            mean_scans=f"{st.mean(x['scans'] for x in res):.1f}",
            mean_recharges=f"{st.mean(x['recharges'] for x in res):.1f}",
            mean_time=f"{st.mean(x['time_used'] for x in res):.0f}",
            stranded=sum(x["stranded"] for x in res),
            wrong_but_confident=sum(1 for x in res if not x["success"] and x["confidence"] >= 0.9),
            seconds=f"{time.time() - t0:.0f}s"))

    with open(os.path.join(args.out, "benchmark.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    cols = list(summary[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(s[c]) for c in cols) + " |" for s in summary]
    text = "\n".join(md)
    with open(os.path.join(args.out, "benchmark.md"), "w") as fh:
        fh.write(f"Benchmark: {args.n} random missions per profile, seeds {args.seed}..{args.seed + args.n - 1}\n\n{text}\n")
    print(text)


if __name__ == "__main__":
    main()
