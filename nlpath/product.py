"""Stage 3: A* on the product graph (cell, k) — plan §2.4c and §3.2.

k = number of sequence elements (w_1, …, w_m, g) entered so far, in order.
Ordering is strict (D1): at progress k the cells of elements k+2 onward are
blocked, so element j+1 is never entered before element j.

Indices in code are 0-based: ``problem.sequence[k]`` is element u_{k+1}, the
next one to enter at progress k.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from nlpath.astar import SearchResult, astar, check_weights, grid_moves, \
    heuristic_scale, octile_field
from nlpath.compiler import CompiledProblem
from nlpath.grid import Cell

State = tuple[Cell, int]


def blocked_masks(problem: CompiledProblem) -> list[np.ndarray]:
    """B_k for k = 0..m+1: union of elements u_{k+2}, …, u_{m+1} (plan §3.2)."""
    n = len(problem.sequence)
    out = []
    for k in range(n + 1):
        mask = np.zeros(problem.allowed.shape, dtype=bool)
        for later in problem.sequence[k + 1:]:
            mask |= later
        out.append(mask)
    return out


def advance(problem: CompiledProblem, cell: Cell, k: int) -> int:
    """adv(u, k): k+1 if ``cell`` is in the next element u_{k+1}, else k."""
    if k < len(problem.sequence) and problem.sequence[k][cell]:
        return k + 1
    return k


def start_problem(problem: CompiledProblem, start: Cell) -> str | None:
    """Why ``start`` is not admissible (plan §3.1), or ``None`` if it is."""
    if not problem.allowed[start]:
        return "start cell is a wall or lies in a forbidden region"
    for j, (name, mask) in enumerate(zip(problem.sequence_names, problem.sequence)):
        if j >= 1 and mask[start]:
            before = problem.sequence_names[j - 1]
            return f"start lies in {name!r}, which must not be entered before {before!r}"
    return None


def plan_product(problem: CompiledProblem, start: Cell) -> SearchResult:
    """Cheapest I-satisfying path from ``start`` (T2–T4), or NO_PATH with a reason.

    The returned ``path`` is the projection of the state path to cells.
    """
    check_weights(problem.allowed, problem.weights)
    start = (int(start[0]), int(start[1]))
    reason = start_problem(problem, start)
    if reason is not None:
        return SearchResult(None, math.inf, 0, reason)

    final = len(problem.sequence)  # m + 1
    blocked = blocked_masks(problem)
    goal_cells = [(int(r), int(c)) for r, c in np.argwhere(problem.goal & problem.allowed)]
    # Heuristic h(v, k) = w_min * octile(v, nearest goal cell); consistent (T4).
    h_field = heuristic_scale(problem.allowed, problem.weights) * \
        octile_field(problem.allowed.shape, goal_cells)

    def successors(state: State) -> Iterator[tuple[State, float]]:
        v, k = state
        # u ∈ N_{S_I}(v) \ B_k: corner rule checked against S_I (plan §3.2).
        for u, step in grid_moves(problem.allowed, v):
            if not blocked[k][u]:
                yield (u, advance(problem, u, k)), step * float(problem.weights[u])

    def is_goal(state: State) -> bool:
        v, k = state
        return k == final and bool(problem.goal[v])

    result = astar((start, advance(problem, start, 0)), is_goal, successors,
                   lambda s: float(h_field[s[0]]))
    if not result.found:
        return SearchResult(None, math.inf, result.nodes_expanded,
                            "no path satisfies the constraints")
    cells = [v for v, _ in result.path]
    return SearchResult(cells, result.cost, result.nodes_expanded)
