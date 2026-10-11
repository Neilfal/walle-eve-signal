"""Small grid / geometry helpers shared by the environment and the agent.

Coordinate system
-----------------
* A cell is ``(x, y)``: ``x`` is the column (0 = left edge), ``y`` is the row (0 = top edge).
* North is *up*, so moving ``N`` decreases ``y``.
* Bearings are compass bearings in degrees, clockwise from North:
  0 = N, 90 = E, 180 = S, 270 = W.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple

Cell = Tuple[int, int]

DIRS: Dict[str, Cell] = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
_DELTA_TO_DIR = {delta: name for name, delta in DIRS.items()}
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def bearing_deg(src: Cell, dst: Cell) -> float:
    """Compass bearing (degrees, clockwise from North) from ``src`` to ``dst``."""
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_diff(a, b):
    """Signed smallest difference ``a - b`` in degrees, in [-180, 180). Works on numpy arrays."""
    return (a - b + 180.0) % 360.0 - 180.0


def compass_word(bearing: float) -> str:
    """Turn a bearing into one of N, NE, E, SE, S, SW, W, NW."""
    return _COMPASS[int(((bearing % 360.0) + 22.5) // 45) % 8]


def euclid(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: Cell, b: Cell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def neighbors4(cell: Cell, width: int, height: int) -> Iterator[Cell]:
    x, y = cell
    for dx, dy in DIRS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield (nx, ny)


def direction_between(a: Cell, b: Cell) -> str:
    """Name of the single move (N/S/E/W) that takes you from ``a`` to the adjacent cell ``b``."""
    return _DELTA_TO_DIR[(b[0] - a[0], b[1] - a[1])]


def bfs(sources: Iterable[Cell], is_free: Callable[[Cell], bool], width: int, height: int):
    """Multi-source 4-connected breadth-first search.

    Returns ``(dist, parent)``. ``dist[c]`` is the number of steps from the nearest source and
    ``parent[c]`` is the neighbour of ``c`` that is one step closer to that source
    (``None`` for the sources themselves).
    """
    dist: Dict[Cell, int] = {}
    parent: Dict[Cell, Optional[Cell]] = {}
    queue = deque()
    for s in sources:
        dist[s] = 0
        parent[s] = None
        queue.append(s)
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur, width, height):
            if nxt not in dist and is_free(nxt):
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                queue.append(nxt)
    return dist, parent


def follow_parents(parent, cell: Cell):
    """Follow parent pointers from ``cell`` back to a BFS source: ``[cell, ..., source]``."""
    path = [cell]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path
