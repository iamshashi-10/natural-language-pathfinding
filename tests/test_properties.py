"""Property-based tests of theorems T1–T5 (plan §3.3, §3.5) — Day 5.

Random small grids (≤ 10×10) with random walls, random region labels (regions
may be disconnected or even empty) and random well-formed intent forms. Starts
are drawn from all cells, so inadmissible starts (walls, forbidden or later
sequence regions) are exercised too.

Ground truth is ``reference_solve``: Dijkstra in networkx on the explicitly
built product graph, written independently of the planner (tests/reference.py).
T6 (correspondence with LTL) needs the M2 planner and is tested on Day 9.
"""

import math

import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from baselines.chaining import plan_chain
from nlpath.astar import grid_moves, heuristic_scale, octile_field
from nlpath.compiler import W_AVOIDED, W_NORMAL, W_PREFERRED, compile_intent
from nlpath.grid import NO_LABEL, REGION_VOCAB, GridMap
from nlpath.intent import IntentForm, validate
from nlpath.pipeline import plan
from nlpath.verify import is_physically_valid, path_length, verify_path
from reference import reference_solve

MANY = settings(max_examples=1000)  # "thousands of random cases" across the file


@st.composite
def problems(draw):
    """(map, well-formed intent, start) on a random labelled grid."""
    h, w = draw(st.integers(2, 10)), draw(st.integers(2, 10))
    n_regions = draw(st.integers(2, 4))
    names = tuple(draw(st.permutations(REGION_VOCAB))[:n_regions])
    cells = h * w
    # ~75% free cells; about half of the free cells are unlabelled, the rest
    # belong to a random region (so regions may be scattered or empty).
    free = np.array(draw(st.lists(st.integers(0, 3), min_size=cells, max_size=cells))) > 0
    codes = np.array(draw(st.lists(st.integers(0, 2 * n_regions - 1),
                                   min_size=cells, max_size=cells)))
    labels = np.where(codes < n_regions, codes, NO_LABEL).astype(np.int8)
    free, labels = free.reshape(h, w), labels.reshape(h, w)
    labels[~free] = NO_LABEL
    gm = GridMap("random", free, labels, names)

    order = draw(st.permutations(names))
    m = draw(st.integers(0, min(2, n_regions - 1)))
    seq, rest = order[:m + 1], order[m + 1:]
    forbidden = tuple(x for x in rest if draw(st.booleans()))
    soft = [draw(st.sampled_from(["none", "preferred", "avoided"])) for _ in names]
    preferred = tuple(n for n, s in zip(names, soft) if s == "preferred")
    avoided = tuple(n for n, s in zip(names, soft) if s == "avoided")
    intent = IntentForm(seq[-1], forbidden, tuple(seq[:-1]), preferred, avoided)
    validate(intent, gm.legend())

    # Mostly a free unlabelled cell (the benchmark's case); sometimes any cell,
    # to exercise the inadmissible-start guard.
    candidates = [tuple(map(int, x)) for x in np.argwhere(free & (labels == NO_LABEL))]
    if candidates and draw(st.integers(0, 9)) > 0:
        start = draw(st.sampled_from(candidates))
    else:
        start = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    return gm, intent, start


@MANY
@given(problems())
def test_T1_to_T4_against_reference(problem):
    """T1 physical validity, T2 soundness, T3 completeness, T4 optimality.

    ``plan`` runs ``verify_path`` on every path it returns and raises on failure,
    so reaching the assertions already establishes T1 and T2 for this case.
    """
    gm, intent, start = problem
    res = plan(gm, intent, start)
    ref = reference_solve(gm, intent, start)
    # Case mix, shown by `pytest --hypothesis-show-statistics`.
    if res.found:
        event(f"path found, {len(intent.ordered_waypoints)} waypoint(s)")
    else:
        event("NO_PATH: inadmissible start" if "start" in res.reason
              else "NO_PATH: infeasible")
    if math.isinf(ref):
        # Soundness of NO_PATH in the other direction: nothing to find.
        assert not res.found
    else:
        assert res.found, "T3: an I-satisfying path exists but NO_PATH was returned"
        assert res.cost == pytest.approx(ref), "T4: cost is not optimal"
        assert is_physically_valid(res.path, gm)          # T1, stated explicitly
        verify_path(res.path, gm, intent, start)          # T2, stated explicitly


@MANY
@given(problems())
def test_T5_bounded_cost_of_compliance(problem):
    """L(path) <= (w_max / w_min) * L*, and = L* when there are no soft preferences."""
    gm, intent, start = problem
    res = plan(gm, intent, start)
    # L*: shortest I_hard-satisfying path; with no soft preferences J equals L.
    l_star = reference_solve(gm, intent.hard_part(), start)
    # Soft preferences never change feasibility.
    assert res.found == (not math.isinf(l_star))
    if not res.found:
        return
    w_min = W_PREFERRED if intent.preferred_regions else W_NORMAL
    w_max = W_AVOIDED if intent.avoided_regions else W_NORMAL
    length = path_length(res.path)
    assert length <= (w_max / w_min) * l_star + 1e-9
    assert length <= 6 * l_star + 1e-9  # the plan's headline constant
    if not intent.preferred_regions and not intent.avoided_regions:
        assert length == pytest.approx(l_star)


@MANY
@given(problems())
def test_chaining_is_sound_and_complete_but_not_better(problem):
    """M3-chain: verified paths, same feasibility as the product search, cost >= optimal."""
    gm, intent, start = problem
    prod = plan(gm, intent, start)
    chain = plan(gm, intent, start, planner=plan_chain)  # verified by the pipeline
    assert chain.found == prod.found
    if prod.found:
        assert chain.cost >= prod.cost - 1e-9
        if intent.ordered_waypoints:
            event("chain strictly worse" if chain.cost > prod.cost + 1e-9
                  else "chain optimal")


@MANY
@given(problems())
def test_heuristic_is_consistent(problem):
    """The assumption behind T4: h(v) <= c(v,u) + h(u) on every edge, h = 0 on goals."""
    gm, intent, _ = problem
    p = compile_intent(gm, intent)
    goal_cells = [tuple(map(int, x)) for x in np.argwhere(p.goal & p.allowed)]
    if not goal_cells:
        return
    h = heuristic_scale(p.allowed, p.weights) * octile_field(gm.shape, goal_cells)
    for v in map(tuple, np.argwhere(p.allowed)):
        if p.goal[v]:
            assert h[v] == 0
        for u, step in grid_moves(p.allowed, v):
            assert h[v] <= step * p.weights[u] + h[u] + 1e-9
