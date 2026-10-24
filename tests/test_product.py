"""Tests for nlpath.product (and the M3-chain ablation) on hand-written intents.

Each hand-checked case states the expected path cost in a comment; every case is
also compared with the independent networkx reference solver.
"""

import math

import numpy as np
import pytest

from baselines.chaining import plan_chain
from nlpath.astar import SQRT2 as R2
from nlpath.compiler import compile_intent
from nlpath.intent import IntentForm
from nlpath.mapgen import benchmark_maps
from nlpath.product import plan_product
from reference import make_map, random_intent, reference_solve


def first_entry(gm, path, name):
    mask = gm.region_mask(name)
    return next((i for i, v in enumerate(path) if mask[v]), None)


def solve(rows, intent, start):
    """Run product search and chaining, and check product against the reference."""
    gm = make_map(rows)
    p = compile_intent(gm, intent)
    prod, chain = plan_product(p, start), plan_chain(p, start)
    ref = reference_solve(gm, intent, start)
    if math.isinf(ref):
        assert not prod.found
    else:
        assert prod.cost == pytest.approx(ref)
    return gm, prod, chain


# ----- forbidden regions -------------------------------------------------------

REROUTE = [
    "#######",
    "#.....#",
    "#.###.#",
    "#.WWWL#",
    "#######",
]


def test_without_prohibition_path_cuts_through_workshop():
    # Straight along row 3: (3,2),(3,3),(3,4) workshop, then (3,5) lab: cost 4.
    gm, prod, _ = solve(REROUTE, IntentForm("lab"), (3, 1))
    assert prod.cost == pytest.approx(4.0)
    assert first_entry(gm, prod.path, "workshop") is not None


def test_prohibition_forces_reroute():
    # Around the top: up 2, right 4, down 2 = 8. The diagonals at both top
    # corners would cut the wall at (2,2)/(2,4), so they are not used.
    gm, prod, chain = solve(REROUTE, IntentForm("lab", ("workshop",)), (3, 1))
    assert prod.cost == pytest.approx(8.0)
    assert first_entry(gm, prod.path, "workshop") is None
    assert chain.cost == pytest.approx(8.0)


# ----- waypoints and strict ordering -------------------------------------------

LINE = [
    "#########",
    "#..LL..S#",
    "#########",
]
LINE_BYPASS = [*LINE[:2], "#.......#", LINE[2]]


def test_goal_in_the_way_of_waypoint_is_infeasible():
    # Strict ordering: the lab may not be entered before the store room, but the
    # only way to the store room is through the lab -> NO_PATH (T3: proof of
    # infeasibility). Without the waypoint the lab is reachable.
    intent = IntentForm("lab", ordered_waypoints=("store_room",))
    _, prod, chain = solve(LINE, intent, (1, 1))
    assert not prod.found and prod.reason == "no path satisfies the constraints"
    assert not chain.found
    _, plain, _ = solve(LINE, IntentForm("lab"), (1, 1))
    assert plain.cost == pytest.approx(2.0)


def test_bypass_around_goal_then_back():
    # To the store room avoiding (1,3),(1,4): (1,2), dip to (2,3),(2,4), back up
    # to (1,5), then (1,6),(1,7): 1 + R2 + 1 + R2 + 1 + 1 = 4 + 2*R2.
    # (Diagonals beside lab cells are fine: lab cells are allowed, just not yet
    # enterable.) Then back to the nearest lab cell (1,4): 3. Total 7 + 2*R2.
    intent = IntentForm("lab", ordered_waypoints=("store_room",))
    gm, prod, chain = solve(LINE_BYPASS, intent, (1, 1))
    assert prod.cost == pytest.approx(7 + 2 * R2)
    assert first_entry(gm, prod.path, "store_room") < first_entry(gm, prod.path, "lab")
    assert chain.cost == pytest.approx(prod.cost)


def test_two_waypoints_and_reentry():
    # Row: P . S . L ; start (1,2); sequence (store_room, print_room, lab).
    # Print room is adjacent but blocked until the store room is visited.
    # Path: S (1), back through (1,2) to P (2), then through S again to L (4) = 7.
    rows = ["#######", "#P.S.L#", "#######"]
    intent = IntentForm("lab", ordered_waypoints=("store_room", "print_room"))
    gm, prod, chain = solve(rows, intent, (1, 2))
    assert prod.path == [(1, 2), (1, 3), (1, 2), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    assert prod.cost == pytest.approx(7.0)
    assert chain.path == prod.path


def test_chaining_is_suboptimal():
    # Waypoint = whole row 1. Chaining enters it at the nearest cell (1,1)
    # (cost 2), then goes to the lab at (3,9): octile 8 cols x 2 rows = 6 + 2*R2.
    # Chain total 8 + 2*R2 = 10.83.
    # Product: two diagonals up to (1,3), along to (1,7), two diagonals down:
    # 4*R2 + 4 = 9.66 — optimal over ALL paths visiting the store room (T4).
    rows = [
        "###########",
        "#SSSSSSSSS#",
        "#.........#",
        "#........L#",
        "###########",
    ]
    intent = IntentForm("lab", ordered_waypoints=("store_room",))
    _, prod, chain = solve(rows, intent, (3, 1))
    assert prod.cost == pytest.approx(4 + 4 * R2)
    assert chain.cost == pytest.approx(8 + 2 * R2)


def test_soft_preference_with_waypoint():
    # Preferred corridor row on top of REROUTE-like map; the prohibition removes
    # the short way, and the corridor (weight 0.5) makes the top route cheaper:
    # (2,1) 1, (1,1)..(1,5) corridor: 0.5 + 4*0.5, (2,5) 1, (3,5) 1 = 5.5.
    rows = ["#######", "#CCCCC#", "#.###.#", "#.WWWL#", "#######"]
    intent = IntentForm("lab", ("workshop",), preferred_regions=("corridor",))
    _, prod, _ = solve(rows, intent, (3, 1))
    assert prod.cost == pytest.approx(5.5)


# ----- start-cell guards (plan §2.5) -------------------------------------------


def test_start_in_forbidden_region():
    _, prod, chain = solve(REROUTE, IntentForm("lab", ("workshop",)), (3, 3))
    assert not prod.found and "forbidden" in prod.reason
    assert chain.reason == prod.reason


def test_start_in_later_sequence_element():
    intent = IntentForm("lab", ordered_waypoints=("store_room",))
    _, prod, _ = solve(LINE_BYPASS, intent, (1, 3))
    assert not prod.found and "'lab'" in prod.reason and "'store_room'" in prod.reason


def test_start_in_first_waypoint_counts_as_visited():
    # Start inside the store room: k starts at 1, so just go to the lab (3 steps).
    intent = IntentForm("lab", ordered_waypoints=("store_room",))
    _, prod, chain = solve(LINE, intent, (1, 7))
    assert prod.cost == pytest.approx(3.0) and chain.cost == pytest.approx(3.0)


def test_start_in_goal_without_waypoints():
    _, prod, chain = solve(LINE, IntentForm("lab"), (1, 3))
    assert prod.path == [(1, 3)] and prod.cost == 0
    assert chain.path == [(1, 3)]


# ----- benchmark maps vs reference ---------------------------------------------


def test_matches_reference_on_benchmark_maps():
    rng = np.random.default_rng(1)
    found = 0
    for gm in benchmark_maps():
        for _ in range(2):
            intent = random_intent(gm, rng)
            p = compile_intent(gm, intent)
            free_unlabelled = np.argwhere(gm.free & (gm.labels == -1))
            start = tuple(map(int, free_unlabelled[rng.integers(len(free_unlabelled))]))
            prod, chain = plan_product(p, start), plan_chain(p, start)
            ref = reference_solve(gm, intent, start)
            if math.isinf(ref):
                assert not prod.found
                continue
            found += 1
            assert prod.cost == pytest.approx(ref)
            if chain.found:  # chaining is sound but never better than optimal
                assert chain.cost >= prod.cost - 1e-9
            seq_entries = [first_entry(gm, prod.path, n) for n in intent.sequence]
            assert None not in seq_entries and seq_entries == sorted(seq_entries)
            assert all(first_entry(gm, prod.path, x) is None
                       for x in intent.forbidden_regions)
    assert found >= 20  # the check is not vacuous
