"""Mission configuration: loading, defaults and validation.

A mission config is a JSON (or YAML) document. See README.md for the full field reference.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import numpy as np

from .geometry import bfs


class ConfigError(ValueError):
    """Raised when a mission config is missing fields or is inconsistent."""


DEFAULT_SENSOR: Dict[str, Any] = {
    "s0": 1.0,                          # signal strength at distance 0
    "decay_length": 8.0,                # strength = s0 * exp(-distance / decay_length)
    "strength_sigma": 0.04,             # std-dev of the strength reading noise
    "bearing_sigma_base_deg": 8.0,      # bearing noise = base + per_cell * distance
    "bearing_sigma_per_cell_deg": 1.5,
    "clue_scan_radius": 2,              # a scan reveals clues within this Chebyshev radius
    "visual_range": 1.5,                # a scan within this distance sees EVE directly
    "spoof_prob": 0.0,                  # chance a scan returns a random (spoofed) signal [hidden]
}

DEFAULT_COSTS: Dict[str, Dict[str, int]] = {
    "move": {"battery": 1, "time": 1},
    "scan": {"battery": 2, "time": 1},
    "recharge": {"battery": 0, "time": 6},
}

CLUE_TYPES = ("bearing", "range", "status")


def load_config(path: str) -> Dict[str, Any]:
    """Load a mission config from a .json, .yaml or .yml file (YAML needs PyYAML)."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML is required for YAML configs: pip install pyyaml") from exc
            return yaml.safe_load(fh)
        return json.load(fh)


def _as_cell(name: str, value) -> tuple:
    try:
        x, y = value
        return int(x), int(y)
    except Exception as exc:
        raise ConfigError(f"'{name}' must be a [x, y] pair, got {value!r}") from exc


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated deep copy of ``raw`` with every default filled in."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping/object")
    cfg = copy.deepcopy(raw)
    for key in ("grid", "start", "chargers", "eve", "initial_signal", "budget"):
        if key not in cfg:
            raise ConfigError(f"missing required key '{key}'")
    cfg.setdefault("name", "mission")
    cfg.setdefault("seed", 0)

    # ---- grid / map -------------------------------------------------------------------
    grid = cfg["grid"]
    if "rows" in grid:
        rows = grid["rows"]
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            raise ConfigError("grid.rows must be a non-empty list of equal-length strings")
        bad = {ch for r in rows for ch in r} - {".", "#"}
        if bad:
            raise ConfigError(f"grid.rows may only contain '.' and '#', found {sorted(bad)}")
        grid["height"], grid["width"] = len(rows), len(rows[0])
        grid["walls"] = [[x, y] for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "#"]
    else:
        for k in ("width", "height"):
            if not isinstance(grid.get(k), int) or grid[k] < 3:
                raise ConfigError(f"grid.{k} must be an integer >= 3")
        grid.setdefault("walls", [])
    width, height = grid["width"], grid["height"]
    walls = {_as_cell("wall", w) for w in grid["walls"]}
    grid["walls"] = sorted([list(w) for w in walls], key=lambda c: (c[1], c[0]))

    def check_cell(name, value):
        cell = _as_cell(name, value)
        if not (0 <= cell[0] < width and 0 <= cell[1] < height):
            raise ConfigError(f"'{name}' {cell} is outside the {width}x{height} grid")
        if cell in walls:
            raise ConfigError(f"'{name}' {cell} is inside a wall")
        return cell

    start = check_cell("start", cfg["start"])
    cfg["start"] = list(start)
    if not cfg["chargers"]:
        raise ConfigError("'chargers' must list at least one charging station")
    cfg["chargers"] = [list(check_cell("charger", c)) for c in cfg["chargers"]]

    eve = cfg["eve"]
    eve["position"] = list(check_cell("eve.position", eve.get("position")))
    eve.setdefault("status", "unknown")

    # ---- sensor / budget / initial signal ----------------------------------------------
    sensor = dict(DEFAULT_SENSOR)
    sensor.update(cfg.get("sensor", {}))
    for key in ("s0", "decay_length", "strength_sigma", "bearing_sigma_base_deg", "visual_range"):
        if not sensor[key] > 0:
            raise ConfigError(f"sensor.{key} must be > 0")
    if not 0 <= sensor["spoof_prob"] <= 1:
        raise ConfigError("sensor.spoof_prob must be in [0, 1]")
    cfg["sensor"] = sensor

    budget = cfg["budget"]
    for key in ("battery", "time"):
        if not isinstance(budget.get(key), (int, float)) or budget[key] <= 0:
            raise ConfigError(f"budget.{key} must be a positive number")
    budget.setdefault("max_recharges", 3)
    costs = copy.deepcopy(DEFAULT_COSTS)
    for action, vals in budget.get("costs", {}).items():
        if action not in costs:
            raise ConfigError(f"unknown action '{action}' in budget.costs")
        costs[action].update(vals)
    budget["costs"] = costs

    sig = cfg["initial_signal"]
    for key in ("bearing_deg", "strength"):
        if key not in sig:
            raise ConfigError(f"initial_signal.{key} is required")
    sig.setdefault("timestamp", 0)
    sig.setdefault("bearing_sigma_deg", 30.0)

    # ---- clues --------------------------------------------------------------------------
    clues = cfg.setdefault("clues", [])
    seen_ids = set()
    for clue in clues:
        for key in ("id", "type", "position"):
            if key not in clue:
                raise ConfigError(f"every clue needs '{key}': {clue!r}")
        if clue["id"] in seen_ids:
            raise ConfigError(f"duplicate clue id '{clue['id']}'")
        seen_ids.add(clue["id"])
        if clue["type"] not in CLUE_TYPES:
            raise ConfigError(f"clue '{clue['id']}': type must be one of {CLUE_TYPES}")
        clue["position"] = list(check_cell(f"clue {clue['id']}", clue["position"]))
        clue.setdefault("decoy", False)
        if clue["type"] == "bearing":
            clue.setdefault("sigma", 15.0)
            if clue["decoy"]:
                mirror = [width - 1 - eve["position"][0], height - 1 - eve["position"][1]]
                clue["decoy_target"] = list(check_cell("decoy_target", clue.get("decoy_target", mirror)))
        elif clue["type"] == "range":
            clue.setdefault("sigma", 1.5)
            clue.setdefault("decoy_offset", 8.0)
        else:
            clue.setdefault("hint", "unknown")

    # ---- everything that matters must be reachable from the start ----------------------
    dist, _ = bfs([start], lambda c: c not in walls, width, height)
    must_reach = [("eve", tuple(eve["position"]))]
    must_reach += [("charger", tuple(c)) for c in cfg["chargers"]]
    must_reach += [(f"clue {c['id']}", tuple(c["position"])) for c in clues]
    for name, cell in must_reach:
        if cell not in dist:
            raise ConfigError(f"{name} at {cell} cannot be reached from the start position")
    return cfg


def build_walls(cfg: Dict[str, Any]) -> np.ndarray:
    """Boolean array of shape (height, width), True where there is a wall (indexed [y, x])."""
    grid = cfg["grid"]
    walls = np.zeros((grid["height"], grid["width"]), dtype=bool)
    for x, y in grid["walls"]:
        walls[y, x] = True
    return walls
