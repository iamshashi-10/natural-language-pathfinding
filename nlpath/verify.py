"""Independent path verifier and instruction checkers (plan §3.5, §4.1).

Written separately from the planner on purpose: this module imports only the
map and intent data types — not ``astar``, ``compiler`` or ``product`` — and
re-derives every rule (legal moves, forbidden cells, strict ordering, weights)
from the definitions in plan §3.1. Agreement between planner and verifier is
therefore evidence, not tautology.

Checkers (plan §4.1) take a path and a map and return ``bool`` — except
``fraction_in_region``, which returns a number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from nlpath.grid import Cell, GridMap
from nlpath.intent import IntentForm

Path = Sequence[Cell]

# Re-stated here (not imported from the compiler) to keep the verifier independent.
WEIGHT_PREFERRED, WEIGHT_NORMAL, WEIGHT_AVOIDED = 0.5, 1.0, 3.0


class PathVerificationError(RuntimeError):
    """A path returned by the pipeline violates a guarantee — always a bug."""


# ----- legal moves -------------------------------------------------------------


def move_problems(path: Path, allowed: np.ndarray) -> list[str]:
    """Why ``path`` is not valid w.r.t. the cell set ``allowed`` (empty if valid).

    Valid (plan §3.1): every cell is in ``allowed``, and each step goes to one
    of the 8 neighbours; a diagonal step also needs both orthogonally adjacent
    cells in ``allowed`` (no corner cutting).
    """
    if len(path) == 0:
        return ["path is empty"]
    h, w = allowed.shape
    problems = []
    for i, (r, c) in enumerate(path):
        if not (0 <= r < h and 0 <= c < w):
            problems.append(f"step {i}: {(r, c)} is outside the map")
        elif not allowed[r, c]:
            problems.append(f"step {i}: {(r, c)} is not an allowed cell")
    if problems:
        return problems
    for i in range(1, len(path)):
        (r0, c0), (r1, c1) = path[i - 1], path[i]
        dr, dc = r1 - r0, c1 - c0
        if (dr, dc) == (0, 0) or max(abs(dr), abs(dc)) > 1:
            problems.append(f"step {i}: {path[i - 1]} -> {path[i]} is not a move "
                            "to a neighbouring cell")
        elif dr and dc and not (allowed[r0, c1] and allowed[r1, c0]):
            problems.append(f"step {i}: diagonal {path[i - 1]} -> {path[i]} cuts a corner")
    return problems


def physical_problems(path: Path, gm: GridMap) -> list[str]:
    """Why ``path`` is not physically valid, i.e. valid w.r.t. free cells (T1)."""
    return move_problems(path, gm.free)


def is_physically_valid(path: Path, gm: GridMap) -> bool:
    return not physical_problems(path, gm)


# ----- checkers (plan §4.1) ----------------------------------------------------


def first_entry(path: Path, gm: GridMap, region: str) -> int | None:
    """e(π): index of the first cell of ``path`` in ``region``, or None."""
    mask = gm.region_mask(region)
    return next((i for i, cell in enumerate(path) if mask[cell]), None)


def ends_in(path: Path, gm: GridMap, region: str) -> bool:
    return len(path) > 0 and bool(gm.region_mask(region)[path[-1]])


def never_enters(path: Path, gm: GridMap, region: str) -> bool:
    return first_entry(path, gm, region) is None


def visits(path: Path, gm: GridMap, region: str) -> bool:
    return first_entry(path, gm, region) is not None


def visits_before(path: Path, gm: GridMap, first: str, then: str) -> bool:
    """Strict ordering (D1): ``first`` is entered, and ``then`` is not entered at
    any step before ``first``'s first entry. ``then`` need not be entered at all.
    """
    e_first = first_entry(path, gm, first)
    if e_first is None:
        return False
    e_then = first_entry(path, gm, then)
    return e_then is None or e_then > e_first


def fraction_in_region(path: Path, gm: GridMap, region: str) -> float:
    """Fraction of path length spent entering cells of ``region``.

    Length-weighted like the cost J: step i counts d(v_i, v_{i+1}) if v_{i+1}
    is in the region. A single-cell path counts 1.0 if that cell is in it.
    """
    mask = gm.region_mask(region)
    if len(path) == 1:
        return float(mask[path[0]])
    total = inside = 0.0
    for a, b in zip(path, path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        total += d
        if mask[b]:
            inside += d
    return inside / total if total else 0.0


CHECKERS = {
    "ends_in": lambda path, gm, c: ends_in(path, gm, c["region"]),
    "never_enters": lambda path, gm, c: never_enters(path, gm, c["region"]),
    "visits": lambda path, gm, c: visits(path, gm, c["region"]),
    "visits_before": lambda path, gm, c: visits_before(path, gm, c["first"], c["then"]),
    "fraction_in_region": lambda path, gm, c: fraction_in_region(path, gm, c["region"]),
}


def run_check(check: dict, path: Path, gm: GridMap) -> bool | float:
    """Evaluate one benchmark check, e.g. ``{"type": "ends_in", "region": "lab"}``."""
    try:
        checker = CHECKERS[check["type"]]
    except KeyError:
        raise ValueError(f"unknown check: {check!r}") from None
    return checker(path, gm, check)


def hard_checks(intent: IntentForm) -> list[dict]:
    """The pass/fail checks that together mean "I_hard-satisfying" (plan §3.1).

    ends_in(g), never_enters(x) for each forbidden x, visits(u_j) for each
    waypoint, and visits_before(u_j, u_{j+1}) for consecutive sequence elements.
    With ends_in(g) these give e_1 < e_2 < … < e_{m+1}.
    """
    seq = intent.sequence
    checks = [{"type": "ends_in", "region": intent.goal_region}]
    checks += [{"type": "never_enters", "region": x} for x in intent.forbidden_regions]
    checks += [{"type": "visits", "region": w} for w in intent.ordered_waypoints]
    checks += [{"type": "visits_before", "first": a, "then": b}
               for a, b in zip(seq, seq[1:])]
    return checks


# ----- cost --------------------------------------------------------------------


def path_length(path: Path) -> float:
    """L(π): sum of step lengths."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def path_cost(path: Path, gm: GridMap, intent: IntentForm) -> float:
    """J(π) = Σ d(v_i, v_{i+1}) · w(v_{i+1}) with the weights of plan §3.1."""
    def weight(cell: Cell) -> float:
        label = gm.label_at(cell)
        if label in intent.preferred_regions:
            return WEIGHT_PREFERRED
        if label in intent.avoided_regions:
            return WEIGHT_AVOIDED
        return WEIGHT_NORMAL
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) * weight(b)
               for a, b in zip(path, path[1:]))


# ----- the runtime verifier ----------------------------------------------------


def verification_problems(path: Path, gm: GridMap, intent: IntentForm,
                          start: Cell) -> list[str]:
    """Every reason ``path`` fails to be I-satisfying from ``start`` (empty if none)."""
    if len(path) == 0:
        return ["path is empty"]
    problems = []
    if tuple(path[0]) != tuple(start):
        problems.append(f"path starts at {path[0]}, not at the start {start}")
    physical = physical_problems(path, gm)
    if physical:
        return problems + physical  # T1 violated; later checks are meaningless
    allowed = gm.free.copy()
    for x in intent.forbidden_regions:
        allowed &= ~gm.region_mask(x)
    problems += move_problems(path, allowed)  # valid w.r.t. S_I (T2)
    for check in hard_checks(intent):
        if not run_check(check, path, gm):
            problems.append(f"failed check {check}")
    return problems


def verify_path(path: Path, gm: GridMap, intent: IntentForm, start: Cell) -> None:
    """Raise ``PathVerificationError`` unless ``path`` is I-satisfying from ``start``."""
    problems = verification_problems(path, gm, intent, start)
    if problems:
        raise PathVerificationError(
            f"path on {gm.map_id} violates intent {intent.to_dict()}: "
            + "; ".join(problems))
