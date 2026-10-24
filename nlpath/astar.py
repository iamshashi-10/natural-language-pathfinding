"""Weighted 8-connected A* (plan §2.5, §3.2).

Two layers:

* ``astar`` — generic A* over hashable states with a closed set and node
  counting. The product-graph planner (Day 3) reuses it with states (cell, k).
* ``plan_grid`` — A* on grid cells: legal moves from ``grid_moves``, edge cost
  ``d(v,u) * w(u)``, heuristic ``w_min * octile(v, nearest goal)``.

Coordinates are ``(row, col)``.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from nlpath.grid import Cell

SQRT2 = math.sqrt(2.0)

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))

S = TypeVar("S", bound=Hashable)


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a search.

    ``path`` is ``None`` when no path exists (NO_PATH); ``reason`` then says why.
    ``nodes_expanded`` counts states popped from the open list and expanded.
    """

    path: list | None
    cost: float
    nodes_expanded: int
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.path is not None


# ----- grid geometry -----------------------------------------------------------


def grid_moves(allowed: np.ndarray, cell: Cell) -> Iterator[tuple[Cell, float]]:
    """Legal moves from ``cell`` within ``allowed``: yields (neighbour, step length).

    A diagonal move is legal only if both orthogonally adjacent cells are
    allowed — no corner cutting (plan §3.1, N_S(v); required by T1).
    """
    h, w = allowed.shape
    r, c = cell
    for dr, dc in ORTHOGONAL:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc]:
            yield (nr, nc), 1.0
    for dr, dc in DIAGONAL:
        nr, nc = r + dr, c + dc
        if (0 <= nr < h and 0 <= nc < w and allowed[nr, nc]
                and allowed[r, nc] and allowed[nr, c]):
            yield (nr, nc), SQRT2


def octile(a: Cell, b: Cell) -> float:
    """Octile distance: shortest step length between cells on an empty 8-grid."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc)


def octile_field(shape: tuple[int, int], goals: Iterable[Cell]) -> np.ndarray:
    """Array of min over goal cells of octile(cell, goal), for every cell.

    ``inf`` everywhere if there are no goals.
    """
    rows, cols = np.indices(shape)
    field = np.full(shape, np.inf)
    for gr, gc in goals:
        dr, dc = np.abs(rows - gr), np.abs(cols - gc)
        d = np.maximum(dr, dc) + (SQRT2 - 1.0) * np.minimum(dr, dc)
        np.minimum(field, d, out=field)
    return field


# ----- generic A* --------------------------------------------------------------


def astar(start: S,
          is_goal: Callable[[S], bool],
          successors: Callable[[S], Iterable[tuple[S, float]]],
          heuristic: Callable[[S], float]) -> SearchResult:
    """A* with a closed set. Returns the optimal path if ``heuristic`` is consistent.

    ``successors(s)`` yields (next_state, edge_cost) with edge_cost > 0.
    Ties on f are broken by insertion order, so results are deterministic.
    """
    counter = itertools.count()
    g: dict[S, float] = {start: 0.0}
    parent: dict[S, S] = {}
    closed: set[S] = set()
    open_heap = [(heuristic(start), next(counter), start)]
    expanded = 0

    while open_heap:
        _, _, s = heapq.heappop(open_heap)
        if s in closed:
            continue  # stale entry; s was already expanded with its optimal g
        if is_goal(s):
            return SearchResult(_reconstruct(parent, s), g[s], expanded)
        closed.add(s)
        expanded += 1
        for t, cost in successors(s):
            if t in closed:
                continue  # with a consistent heuristic, closed g is optimal
            new_g = g[s] + cost
            if new_g < g.get(t, math.inf):
                g[t] = new_g
                parent[t] = s
                heapq.heappush(open_heap, (new_g + heuristic(t), next(counter), t))
    return SearchResult(None, math.inf, expanded, "goal unreachable")


def _reconstruct(parent: dict, s) -> list:
    path = [s]
    while s in parent:
        s = parent[s]
        path.append(s)
    path.reverse()
    return path


# ----- A* on the grid ----------------------------------------------------------


def heuristic_scale(allowed: np.ndarray, weights: np.ndarray) -> float:
    """w_min: the smallest weight of any cell a path can enter.

    Every edge costs at least w_min * step length, so w_min * octile is a lower
    bound on remaining cost — admissible and consistent (T4). With plan weights
    this is 0.5 when a reachable preferred cell exists, else 1.0 (or 3.0).
    """
    if not allowed.any():
        return 1.0
    return float(weights[allowed].min())


def check_weights(allowed: np.ndarray, weights: np.ndarray) -> None:
    """Edge costs must be positive (plan §3 invariant; required by T4)."""
    if weights.shape != allowed.shape:
        raise ValueError("weights and allowed must have the same shape")
    if allowed.any() and not (weights[allowed] > 0).all():
        raise ValueError("cell weights must be positive on allowed cells")


def plan_grid(allowed: np.ndarray, weights: np.ndarray, start: Cell,
              goals: np.ndarray, blocked: np.ndarray | None = None) -> SearchResult:
    """Cheapest path from ``start`` to any cell of ``goals`` within ``allowed``.

    Args:
        allowed: bool (H, W); cells the path may occupy (free and not forbidden).
        weights: float (H, W); cost multiplier of entering each cell.
        start: start cell.
        goals: bool (H, W) mask of goal cells (only allowed ones count).
        blocked: optional bool (H, W); allowed cells that may not be entered.
            Unlike removing them from ``allowed``, they still count for the
            corner rule — the same move set as the product search (plan §3.2).

    Edge cost is ``d(v,u) * weights[u]`` (plan §2.5). Returns NO_PATH with a
    reason if the start is not allowed or no goal cell is reachable.
    """
    check_weights(allowed, weights)
    start = (int(start[0]), int(start[1]))
    if not allowed[start]:
        return SearchResult(None, math.inf, 0, "start cell is not allowed")
    if blocked is None:
        blocked = np.zeros(allowed.shape, dtype=bool)
    enterable = allowed & ~blocked
    goal_mask = goals & enterable
    goal_cells = [(int(r), int(c)) for r, c in np.argwhere(goal_mask)]
    if not goal_cells:
        return SearchResult(None, math.inf, 0, "no allowed goal cell")

    w_min = heuristic_scale(allowed, weights)
    h_field = w_min * octile_field(allowed.shape, goal_cells)

    def successors(v: Cell) -> Iterator[tuple[Cell, float]]:
        for u, step in grid_moves(allowed, v):
            if enterable[u]:
                yield u, step * float(weights[u])

    return astar(start, lambda v: bool(goal_mask[v]), successors,
                 lambda v: float(h_field[v]))
