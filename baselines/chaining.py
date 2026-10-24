"""M3-chain ablation: sequential waypoint chaining instead of the product graph.

Plans start → w_1 → … → w_m → g as separate A* searches and concatenates them
(plan §2.4c). Each segment j is searched with the cells of later sequence
elements removed, so the result obeys strict ordering and is I-satisfying —
but each segment fixes its entry cell greedily, so the total is not optimal in
general. Measuring that gap is the point of this ablation.
"""

from __future__ import annotations

import math

from nlpath.astar import SearchResult, plan_grid
from nlpath.compiler import CompiledProblem
from nlpath.grid import Cell
from nlpath.product import advance, blocked_masks, start_problem


def plan_chain(problem: CompiledProblem, start: Cell) -> SearchResult:
    """Chain one A* search per remaining sequence element; NO_PATH if any fails."""
    start = (int(start[0]), int(start[1]))
    reason = start_problem(problem, start)
    if reason is not None:
        return SearchResult(None, math.inf, 0, reason)

    blocked = blocked_masks(problem)
    k = advance(problem, start, 0)  # start may already be in the first waypoint
    path, cost, expanded = [start], 0.0, 0
    while k < len(problem.sequence):
        allowed = problem.allowed & ~blocked[k]
        seg = plan_grid(allowed, problem.weights, path[-1], problem.sequence[k])
        expanded += seg.nodes_expanded
        if not seg.found:
            name = problem.sequence_names[k]
            return SearchResult(None, math.inf, expanded, f"no path to {name!r}")
        path += seg.path[1:]
        cost += seg.cost
        k += 1
    return SearchResult(path, cost, expanded)
