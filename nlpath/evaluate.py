"""Scoring one method output against one benchmark instruction (plan §4.3).

Every method's output is judged the same way, against the GOLD intent, using
the independent checkers in ``nlpath.verify``.
"""

from __future__ import annotations

from collections.abc import Sequence

from nlpath.benchmark import Instruction
from nlpath.grid import Cell, GridMap
from nlpath.verify import (path_cost, path_length, physical_problems, run_check,
                           verification_problems)


def score(gm: GridMap, it: Instruction, path: Sequence[Cell] | None) -> dict:
    """Metrics for ``path`` (None means the method answered NO_PATH or failed).

    * ``valid``: physically walkable, starts at the start, ends in the goal
      region — the "path validity" metric.
    * ``hard_ok``: satisfies every hard constraint of the gold intent.
    * ``correct``: hard_ok on feasible instructions; NO_PATH on infeasible ones.
    """
    goal = it.gold_intent.goal_region
    row: dict = {"returned_path": path is not None}
    if path is None:
        row.update(valid=False, hard_ok=False, correct=not it.feasible,
                   problems=[], checks={}, length=None, cost=None, cost_ratio=None)
        return row
    path = [tuple(cell) for cell in path]
    walk = physical_problems(path, gm)
    in_map = not walk or all("outside the map" not in p for p in walk)
    ends = in_map and len(path) > 0 and gm.in_bounds(path[-1]) and \
        bool(gm.region_mask(goal)[path[-1]])
    valid = not walk and path[0] == tuple(it.start) and ends
    problems = verification_problems(path, gm, it.gold_intent, it.start)
    checks = {}
    if not walk:
        for check in it.checks:
            key = check["type"] + ":" + ",".join(
                str(v) for k, v in check.items() if k != "type")
            checks[key] = run_check(check, path, gm)
    cost = path_cost(path, gm, it.gold_intent) if not walk else None
    row.update(valid=valid, hard_ok=not problems, correct=not problems and it.feasible,
               problems=problems[:5], checks=checks,
               length=path_length(path) if not walk else None, cost=cost,
               cost_ratio=(cost / it.ref_cost) if cost is not None and it.ref_cost else None)
    return row
