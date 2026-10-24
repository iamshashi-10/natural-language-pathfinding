"""M3-chain ablation: sequential waypoint chaining instead of the product graph.

Plans start → w_1 → … → w_m → g as separate A* searches and concatenates them
(plan §2.4c). Each segment j may not enter cells of later sequence elements,
so the result obeys strict ordering and is I-satisfying — but each segment
fixes its entry cell greedily, so the total is not optimal in general.
Measuring that gap is the point of this ablation.

It is complete, though: the blocked set only shrinks as k grows and moves are
symmetric, so from the entry cell of u_j the walk back along the path so far,
followed by any I-satisfying path, reaches u_{j+1}.
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
        # Same move set as the product search: later elements cannot be entered
        # but still count as allowed for the corner rule (plan §3.2).
        seg = plan_grid(problem.allowed, problem.weights, path[-1],
                        problem.sequence[k], blocked=blocked[k])
        expanded += seg.nodes_expanded
        if not seg.found:
            name = problem.sequence_names[k]
            return SearchResult(None, math.inf, expanded, f"no path to {name!r}")
        path += seg.path[1:]
        cost += seg.cost
        k += 1
    return SearchResult(path, cost, expanded)
