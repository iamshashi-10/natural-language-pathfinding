"""Tests for nlpath.verify on known-good and known-bad paths (Day 4)."""

import math

import pytest

from nlpath.intent import IntentForm
from nlpath.verify import (PathVerificationError, ends_in, first_entry,
                           fraction_in_region, hard_checks, is_physically_valid,
                           never_enters, path_cost, path_length, physical_problems,
                           run_check, verification_problems, verify_path, visits,
                           visits_before)
from reference import make_map

R2 = math.sqrt(2)

#        col 0123456789
ROWS = ["##########",   # 0
        "#..WW....#",   # 1   W = workshop
        "#...#....#",   # 2   (2,4) is a wall
        "#S.....LL#",   # 3   S = store_room, L = lab
        "##########"]   # 4
GM = make_map(ROWS)

# Over the top through the workshop to the lab. Diagonals checked by hand:
# (3,2)->(2,3) needs (2,2),(3,3); (1,5)->(2,6) needs (1,6),(2,5);
# (2,6)->(3,7) needs (2,7),(3,6) — all free.
THROUGH_W = [(3, 2), (2, 3), (1, 3), (1, 4), (1, 5), (2, 6), (3, 7)]
# Store room first, then the lab along row 3.
VIA_S = [(3, 2), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
# Lab first, then the store room, then the lab again (ordering violated).
LAB_FIRST = [(3, 6), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1),
             (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
# Passes diagonally by the workshop's corner at (1,3) without entering it:
# (1,2)->(2,3) has orthogonals (1,3)=W and (2,2). Physically valid; invalid
# once the workshop is forbidden (corner rule w.r.t. S_I, plan §3.1).
PAST_W_CORNER = [(3, 2), (2, 2), (1, 2), (2, 3), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]

AVOID_W = IntentForm("lab", forbidden_regions=("workshop",))
S_THEN_LAB = IntentForm("lab", ordered_waypoints=("store_room",))


# ----- physical validity (T1) --------------------------------------------------


@pytest.mark.parametrize("path", [THROUGH_W, VIA_S, LAB_FIRST, PAST_W_CORNER, [(3, 2)]])
def test_known_good_paths_are_physically_valid(path):
    assert is_physically_valid(path, GM)


@pytest.mark.parametrize("path, expected", [
    ([], "path is empty"),
    ([(3, 3), (2, 4)], "step 1: (2, 4) is not an allowed cell"),          # wall
    ([(0, 0)], "step 0: (0, 0) is not an allowed cell"),                  # border wall
    ([(3, 2), (3, 4)], "step 1: (3, 2) -> (3, 4) is not a move to a neighbouring cell"),
    ([(3, 2), (3, 2)], "step 1: (3, 2) -> (3, 2) is not a move to a neighbouring cell"),
    ([(3, 2), (3, -1)], "step 1: (3, -1) is outside the map"),
    # Orthogonals of (2,3)->(1,4) are (1,3) free and (2,4) wall.
    ([(2, 3), (1, 4)], "step 1: diagonal (2, 3) -> (1, 4) cuts a corner"),
])
def test_known_bad_paths(path, expected):
    assert physical_problems(path, GM) == [expected]


# ----- checkers ----------------------------------------------------------------


def test_first_entry_and_simple_checkers():
    assert first_entry(THROUGH_W, GM, "workshop") == 2
    assert first_entry(VIA_S, GM, "workshop") is None
    assert ends_in(VIA_S, GM, "lab") and not ends_in(VIA_S[:-1], GM, "lab")
    assert not ends_in([], GM, "lab")
    assert never_enters(VIA_S, GM, "workshop") and not never_enters(THROUGH_W, GM, "workshop")
    assert visits(VIA_S, GM, "store_room") and not visits(THROUGH_W, GM, "store_room")


def test_visits_before_is_strict():
    assert visits_before(VIA_S, GM, "store_room", "lab")
    # Lab entered at step 1, store room only at step 7.
    assert not visits_before(LAB_FIRST, GM, "store_room", "lab")
    # `first` never entered -> False.
    assert not visits_before(THROUGH_W, GM, "store_room", "lab")
    # `then` never entered -> True (nothing was entered too early).
    assert visits_before(VIA_S, GM, "store_room", "workshop")


def test_fraction_in_region():
    # VIA_S: 7 unit steps, only the last enters the lab.
    assert fraction_in_region(VIA_S, GM, "lab") == pytest.approx(1 / 7)
    # THROUGH_W: steps √2, 1(W), 1(W), 1, √2, √2 -> 2 / (3 + 3√2).
    assert fraction_in_region(THROUGH_W, GM, "workshop") == pytest.approx(2 / (3 + 3 * R2))
    assert fraction_in_region([(3, 7)], GM, "lab") == 1.0
    assert fraction_in_region([(3, 2)], GM, "lab") == 0.0


def test_run_check_dispatch():
    assert run_check({"type": "ends_in", "region": "lab"}, VIA_S, GM) is True
    assert run_check({"type": "visits_before", "first": "store_room", "then": "lab"},
                     LAB_FIRST, GM) is False
    assert run_check({"type": "fraction_in_region", "region": "lab"},
                     VIA_S, GM) == pytest.approx(1 / 7)
    with pytest.raises(ValueError, match="unknown check"):
        run_check({"type": "teleports"}, VIA_S, GM)


def test_hard_checks():
    intent = IntentForm("lab", ("workshop",), ("store_room", "print_room"))
    assert hard_checks(intent) == [
        {"type": "ends_in", "region": "lab"},
        {"type": "never_enters", "region": "workshop"},
        {"type": "visits", "region": "store_room"},
        {"type": "visits", "region": "print_room"},
        {"type": "visits_before", "first": "store_room", "then": "print_room"},
        {"type": "visits_before", "first": "print_room", "then": "lab"},
    ]


# ----- length and cost ---------------------------------------------------------


def test_path_length_and_cost():
    assert path_length([(3, 2)]) == 0
    assert path_length(THROUGH_W) == pytest.approx(3 + 3 * R2)
    assert path_cost(THROUGH_W, GM, IntentForm("lab")) == pytest.approx(3 + 3 * R2)
    # Avoided workshop: the two W steps cost 3 each -> √2 + 3 + 3 + 1 + √2 + √2.
    avoid = IntentForm("lab", avoided_regions=("workshop",))
    assert path_cost(THROUGH_W, GM, avoid) == pytest.approx(7 + 3 * R2)
    # Preferred lab: the last step (diagonal into (3,7)) costs 0.5·√2.
    prefer = IntentForm("lab", preferred_regions=("lab",))
    assert path_cost(THROUGH_W, GM, prefer) == pytest.approx(3 + 2.5 * R2)


# ----- the verifier ------------------------------------------------------------


@pytest.mark.parametrize("path, intent, start", [
    (THROUGH_W, IntentForm("lab"), (3, 2)),
    (VIA_S, AVOID_W, (3, 2)),
    (VIA_S, S_THEN_LAB, (3, 2)),
    (PAST_W_CORNER, IntentForm("lab"), (3, 2)),
    ([(3, 7)], IntentForm("lab"), (3, 7)),
])
def test_verify_accepts_satisfying_paths(path, intent, start):
    verify_path(path, GM, intent, start)


@pytest.mark.parametrize("path, intent, start, expected", [
    (THROUGH_W, AVOID_W, (3, 2), [
        "step 2: (1, 3) is not an allowed cell",
        "step 3: (1, 4) is not an allowed cell",
        "failed check {'type': 'never_enters', 'region': 'workshop'}",
    ]),
    (PAST_W_CORNER, AVOID_W, (3, 2), [
        "step 3: diagonal (1, 2) -> (2, 3) cuts a corner",
    ]),
    (THROUGH_W, S_THEN_LAB, (3, 2), [
        "failed check {'type': 'visits', 'region': 'store_room'}",
        "failed check {'type': 'visits_before', 'first': 'store_room', 'then': 'lab'}",
    ]),
    (LAB_FIRST, S_THEN_LAB, (3, 6), [
        "failed check {'type': 'visits_before', 'first': 'store_room', 'then': 'lab'}",
    ]),
    (VIA_S[:-1], IntentForm("lab"), (3, 2), [
        "failed check {'type': 'ends_in', 'region': 'lab'}",
    ]),
    (VIA_S, IntentForm("lab"), (3, 3), [
        "path starts at (3, 2), not at the start (3, 3)",
    ]),
    ([(3, 2), (2, 4), (3, 6)], IntentForm("lab"), (3, 2), [
        "step 1: (2, 4) is not an allowed cell",   # physical failure stops further checks
    ]),
])
def test_verify_rejects_violations(path, intent, start, expected):
    assert verification_problems(path, GM, intent, start) == expected
    with pytest.raises(PathVerificationError):
        verify_path(path, GM, intent, start)
