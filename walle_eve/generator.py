"""Random mission generator, used for the generalisation benchmark and for making new configs."""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .config import ConfigError, DEFAULT_SENSOR, normalize_config
from .geometry import bearing_deg, euclid

STATUSES = ["active", "low_power", "damaged", "dormant"]


def random_config(seed: int, width: int = 15, height: int = 15, wall_density: float = 0.12,
                  n_clues: int = 4, n_decoys: int = 0, spoof_prob: float = 0.0, noise: float = 1.0,
                  battery: int = 40, time_budget: int = 160, max_recharges: int = 3,
                  second_charger: bool = False) -> Dict[str, Any]:
    """Build a random, valid mission config. ``noise`` scales every sensor noise level."""
    rng = np.random.default_rng(seed)
    sensor = dict(DEFAULT_SENSOR)
    sensor["strength_sigma"] *= noise
    sensor["bearing_sigma_base_deg"] *= noise
    sensor["bearing_sigma_per_cell_deg"] *= noise
    sensor["spoof_prob"] = spoof_prob
    start = (1, 1)
    chargers = [start] + ([(width - 2, height - 2)] if second_charger else [])
    for _ in range(500):
        walls = rng.random((height, width)) < wall_density
        free = [(x, y) for y in range(height) for x in range(width) if not walls[y, x]]
        far = [c for c in free if euclid(c, start) >= 0.55 * min(width, height) and c not in chargers]
        if len(far) < 5 or len(free) < n_clues + 10:
            continue
        eve = far[int(rng.integers(len(far)))]
        pool = [c for c in free if c not in chargers and c != eve]
        picks = [pool[i] for i in rng.choice(len(pool), size=n_clues, replace=False)]
        status = STATUSES[int(rng.integers(len(STATUSES)))]
        kinds = ["bearing", "range", "status", "bearing", "range", "bearing"]
        clues = []
        for i, pos in enumerate(picks):
            kind = kinds[i % len(kinds)]
            clue: Dict[str, Any] = {"id": f"c{i + 1}", "type": kind, "position": list(pos)}
            if kind == "bearing":
                clue["sigma"] = 15.0 * noise
            elif kind == "range":
                clue["sigma"] = 1.5 * noise
            else:
                clue["hint"] = status
            clues.append(clue)
        decoyable = [c for c in clues if c["type"] != "status"]
        for c in decoyable[:n_decoys]:
            c["decoy"] = True
            if c["type"] == "bearing":
                c["decoy_target"] = list(free[int(rng.integers(len(free)))])
            else:
                c["decoy_offset"] = float(rng.choice([6.0, 9.0, 12.0]))
        d = euclid(start, eve)
        init_bearing = (bearing_deg(start, eve) + rng.normal(0, 30)) % 360
        init_strength = max(0.0, math.exp(-d / sensor["decay_length"]) + rng.normal(0, 2 * sensor["strength_sigma"]))
        cfg = {
            "name": f"random mission (seed {seed})",
            "seed": int(seed),
            "grid": {"width": width, "height": height,
                     "walls": [[x, y] for y in range(height) for x in range(width)
                               if walls[y, x] and (x, y) not in chargers and (x, y) != eve
                               and (x, y) not in picks]},
            "start": list(start),
            "chargers": [list(c) for c in chargers],
            "eve": {"position": list(eve), "status": status},
            "initial_signal": {"bearing_deg": round(float(init_bearing), 1), "strength": round(float(init_strength), 3),
                               "timestamp": 0, "bearing_sigma_deg": 30},
            "budget": {"battery": battery, "time": time_budget, "max_recharges": max_recharges},
            "sensor": sensor,
            "clues": clues,
        }
        try:
            return normalize_config(cfg) and cfg
        except ConfigError:
            continue
    raise RuntimeError("could not generate a valid mission; try a lower wall_density")
