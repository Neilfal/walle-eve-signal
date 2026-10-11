"""Bayesian belief over where EVE is: one probability per free grid cell.

Every observation is turned into a *likelihood map* (how probable is this reading if EVE were
in cell c?). The posterior is ``prior x likelihood``, renormalised. All likelihoods are mixed
with a small uniform "outlier" component so a spoofed signal or a decoy clue can dent the
belief but never destroy it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .geometry import angle_diff

SQRT_2PI = np.sqrt(2.0 * np.pi)


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * SQRT_2PI)


# --------------------------------------------------------------------------- sensor models
def expected_strength(sensor: Dict, dist):
    return sensor["s0"] * np.exp(-dist / sensor["decay_length"])


def bearing_sigma(sensor: Dict, dist):
    return sensor["bearing_sigma_base_deg"] + sensor["bearing_sigma_per_cell_deg"] * dist


def strength_likelihood(sensor: Dict, obs, dist, eps: float, sigma=None):
    sigma = sensor["strength_sigma"] if sigma is None else sigma
    return (1 - eps) * gauss(obs, expected_strength(sensor, dist), sigma) + eps * 1.0  # U[0,1]


def bearing_likelihood(obs, true_bearing, sigma, eps: float, dist=None):
    lik = (1 - eps) * gauss(angle_diff(obs, true_bearing), 0.0, sigma) + eps / 360.0
    if dist is not None:  # standing on the cell: bearing is meaningless, say nothing
        lik = np.where(dist < 0.5, 1.0 / 360.0, lik)
    return lik


def range_likelihood(obs, dist, sigma, eps: float, max_dist: float):
    return (1 - eps) * gauss(obs, dist, sigma) + eps / max_dist


class Geometry:
    """Caches, for any cell, the distance and bearing from it to every other cell."""

    def __init__(self, width: int, height: int):
        ys, xs = np.mgrid[0:height, 0:width]
        self.X, self.Y = xs.astype(float), ys.astype(float)
        self.max_dist = float(np.hypot(width, height))
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def dist_bearing(self, cell):
        cell = (int(cell[0]), int(cell[1]))
        if cell not in self._cache:
            dx, dy = self.X - cell[0], self.Y - cell[1]
            dist = np.hypot(dx, dy)
            bearing = np.degrees(np.arctan2(dx, -dy)) % 360.0
            self._cache[cell] = (dist, bearing)
        return self._cache[cell]


class Belief:
    def __init__(self, walls: np.ndarray, sensor: Dict, outlier_prob=0.05, clue_outlier_prob=0.2):
        self.height, self.width = walls.shape
        self.free = ~walls
        self.sensor = sensor
        self.eps = outlier_prob
        self.clue_eps = clue_outlier_prob
        self.geo = Geometry(self.width, self.height)
        self.p = self.free / self.free.sum()          # uniform prior over free cells

    # ------------------------------------------------------------------ updates
    def _apply(self, likelihood):
        post = self.p * likelihood * self.free
        total = post.sum()
        if not np.isfinite(total) or total <= 0:      # should not happen thanks to outlier floors
            post, total = self.free.astype(float), self.free.sum()
        self.p = post / total

    def update_strength(self, pos, obs, sigma=None):
        dist, _ = self.geo.dist_bearing(pos)
        self._apply(strength_likelihood(self.sensor, obs, dist, self.eps, sigma))

    def update_bearing(self, pos, obs, sigma=None):
        """Bearing reading taken at ``pos``. Default noise grows with distance (sensor model)."""
        dist, bear = self.geo.dist_bearing(pos)
        sig = bearing_sigma(self.sensor, dist) if sigma is None else sigma
        self._apply(bearing_likelihood(obs, bear, sig, self.eps, dist))

    def update_clue_bearing(self, clue_pos, obs, sigma):
        dist, bear = self.geo.dist_bearing(clue_pos)
        self._apply(bearing_likelihood(obs, bear, sigma, self.clue_eps, dist))

    def update_clue_range(self, clue_pos, obs, sigma):
        dist, _ = self.geo.dist_bearing(clue_pos)
        self._apply(range_likelihood(obs, dist, sigma, self.clue_eps, self.geo.max_dist))

    def update_visual(self, pos, seen: bool, eve_pos=None):
        """A scan within visual range either sees EVE (belief collapses) or rules out nearby cells."""
        if seen:
            p = np.zeros_like(self.p)
            p[int(eve_pos[1]), int(eve_pos[0])] = 1.0
            self.p = p
            return
        dist, _ = self.geo.dist_bearing(pos)
        self._apply(np.where(dist <= self.sensor["visual_range"], 0.01, 1.0))

    # ------------------------------------------------------------------ summaries
    def entropy(self) -> float:
        p = self.p[self.p > 0]
        return float(-(p * np.log2(p)).sum())

    def _smoothed(self) -> np.ndarray:
        """Probability mass inside the 3x3 block centred on every cell."""
        padded = np.pad(self.p, 1)
        h, w = self.p.shape
        return sum(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    def estimate(self) -> Tuple[int, int]:
        """Best guess for EVE's cell: the centre of the 3x3 block holding the most probability."""
        # tiny tie-break towards the single most likely cell inside the winning block
        s = np.where(self.free, self._smoothed() + 1e-3 * self.p, -1.0)
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        return int(x), int(y)

    def confidence(self) -> float:
        """Probability that EVE is within one cell (3x3 block) of ``estimate()``."""
        x, y = self.estimate()
        return float(self._smoothed()[y, x])

    def credible_cells(self, mass: float = 0.9) -> int:
        """Smallest number of cells that together hold ``mass`` of the probability."""
        flat = np.sort(self.p.ravel())[::-1]
        return int(np.searchsorted(np.cumsum(flat), mass) + 1)

    def top_cells(self, n: int = 5):
        idx = np.argsort(self.p.ravel())[::-1][:n]
        return [((int(i % self.width), int(i // self.width)), float(self.p.ravel()[i])) for i in idx]

    def sample(self, rng: np.random.Generator, k: int):
        """Low-variance (systematic) sample of k cells from the belief. Returns (ys, xs)."""
        cdf = np.cumsum(self.p.ravel())
        cdf[-1] = 1.0
        u = (rng.random() + np.arange(k)) / k
        idx = np.minimum(np.searchsorted(cdf, u), cdf.size - 1)
        return np.unravel_index(idx, self.p.shape)
