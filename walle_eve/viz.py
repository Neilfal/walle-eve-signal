"""Visualisations: ASCII map frames (always) and a PNG summary (if matplotlib is installed)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

LEGEND = ("Legend: # wall | B charger | W WALL-E | * path | c clue found | X inferred EVE | E true EVE (eval only) | "
          "digits 1-9 = belief relative to peak ('.' = ~0)")


def render_ascii(walls: np.ndarray, chargers: Iterable[Tuple[int, int]], frame: Dict[str, Any],
                 truth: Optional[Tuple[int, int]] = None, title: str = "") -> str:
    h, w = walls.shape
    p = frame["p"]
    peak = p.max() if p.max() > 0 else 1.0
    grid = [["#" if walls[y, x] else ("." if p[y, x] / peak < 0.02 else str(min(9, max(1, int(10 * p[y, x] / peak)))))
             for x in range(w)] for y in range(h)]
    for (x, y) in frame["path"]:
        grid[y][x] = "*"
    for (x, y) in frame["clues"].values():
        grid[y][x] = "c"
    for (x, y) in chargers:
        grid[y][x] = "B"
    ex, ey = frame["estimate"]
    grid[ey][ex] = "X"
    if truth is not None:
        grid[truth[1]][truth[0]] = "E"
    px, py = frame["position"]
    grid[py][px] = "W"
    header = "    " + "".join(str(x % 10) for x in range(w))
    rows = [f"{y:3d} " + "".join(grid[y]) for y in range(h)]
    return "\n".join(([title] if title else []) + [header] + rows)


def ascii_frames(agent, env, truth_at_end: bool = True) -> str:
    out = [LEGEND, ""]
    last = len(agent.frames) - 1
    for i, fr in enumerate(agent.frames):
        label = "start (initial signal only)" if fr["step"] == 0 else f"after scan at step {fr['step']}"
        truth = tuple(env.ground_truth()["eve_position"]) if (truth_at_end and i == last) else None
        out.append(render_ascii(env.walls, env.chargers, fr, truth, title=f"--- frame {i}: {label} ---"))
        out.append("")
    return "\n".join(out)


def plot_summary(agent, env, path: str) -> bool:
    """PNG with the final belief + path, and entropy / confidence / battery over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    truth = env.ground_truth()
    fr = agent.frames[-1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    belief = np.ma.masked_where(env.walls, fr["p"])
    ax.imshow(np.where(env.walls, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.6)
    im = ax.imshow(belief, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(EVE in cell)")
    px, py = zip(*fr["path"])
    ax.plot(px, py, "-", color="white", lw=1.4, alpha=0.9, label="WALL-E path")
    sx, sy = zip(*agent.scan_log) if agent.scan_log else ([], [])
    ax.scatter(sx, sy, marker="o", s=38, facecolors="none", edgecolors="orange", label="scan")
    for c in truth["clues"]:
        x, y = c["position"]
        if c["discovered"]:
            ax.scatter([x], [y], marker="*", s=140, color="gold", edgecolors="k", zorder=5)
        else:
            ax.scatter([x], [y], marker="*", s=90, facecolors="none", edgecolors="gray", zorder=4)
    ax.scatter([], [], marker="*", s=110, color="gold", edgecolors="k", label="clue found")
    ax.scatter([], [], marker="*", s=80, facecolors="none", edgecolors="gray", label="clue never found")
    ax.scatter([], [], marker="s", s=70, color="deepskyblue", edgecolors="k", label="charger")
    for (x, y) in env.chargers:
        ax.scatter([x], [y], marker="s", s=90, color="deepskyblue", edgecolors="k", zorder=5)
    ex, ey = truth["eve_position"]
    ax.scatter([ex], [ey], marker="X", s=170, color="red", edgecolors="white", zorder=6, label="true EVE")
    ax.scatter([fr["estimate"][0]], [fr["estimate"][1]], marker="+", s=220, color="cyan", lw=2.5, zorder=7,
               label="estimate")
    ax.set_title("Final belief, path, scans and clues")
    ax.set_xticks(range(0, env.width, 2)); ax.set_yticks(range(0, env.height, 2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=8, frameon=False)

    steps = [h["step"] for h in agent.history]
    ax = axes[1]
    ax.plot(steps, [h["entropy"] for h in agent.history], color="tab:purple", label="entropy (bits)")
    ax.set_xlabel("environment step"); ax.set_ylabel("belief entropy (bits)", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(steps, [h["confidence"] for h in agent.history], color="tab:green", label="confidence")
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("confidence", color="tab:green")
    ax.set_title("Uncertainty shrinks as evidence arrives")

    ax = axes[2]
    ax.plot(steps, [h["battery"] for h in agent.history], color="tab:red")
    ax.set_ylim(0, agent.max_battery * 1.05)
    ax.set_xlabel("environment step"); ax.set_ylabel("battery")
    for t in agent.strategy_timeline:
        ax.axvline(min(steps[-1], max(0, t["action_no"])), color="gray", ls=":", lw=0.8)
    ax.set_title("Battery (dotted: strategy changes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
