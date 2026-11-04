"""Tests for nlpath.errors: each failure gets the intended stage and type."""

import pytest

from nlpath.errors import classify

GOLD = {"goal_region": "lab", "forbidden_regions": ["workshop"],
        "ordered_waypoints": ["store_room", "office"], "preferred_regions": [],
        "avoided_regions": ["atrium"]}


def m3(parsed, correct=False, exact=False):
    return {"method": "M3", "correct": correct, "parsed": parsed, "parse_exact": exact}


def form(**changes):
    return {**GOLD, **changes}


@pytest.mark.parametrize("parsed, expected", [
    (None, "no valid intent form after 3 calls"),
    (form(goal_region="office"), "wrong goal region"),
    (form(ordered_waypoints=["office", "store_room"]), "ordering inverted"),
    (form(ordered_waypoints=["store_room"]), "waypoint / ordering dropped"),
    (form(forbidden_regions=["workshop", "atrium"], avoided_regions=[]),
     "soft preference read as prohibition"),
    (form(forbidden_regions=[], avoided_regions=["atrium", "workshop"]),
     "prohibition read as soft preference"),
    (form(forbidden_regions=[]), "prohibition missed"),
    (form(avoided_regions=[]), "soft preference missed"),
])
def test_m3_interpretation_errors(parsed, expected):
    assert classify(m3(parsed), GOLD, True) == ("interpretation", expected)


def test_m3_correct_and_bug():
    assert classify(m3(GOLD, correct=True, exact=True), GOLD, True) == ("ok", "correct")
    assert classify(m3(GOLD, exact=True), GOLD, True)[0] == "planning"


def m1(path, problems=(), correct=False, fmt=False):
    return {"method": "M1", "correct": correct, "path": path, "problems": list(problems),
            "format_failure": fmt, "returned_path": path is not None, "valid": False}


@pytest.mark.parametrize("row, feasible, expected", [
    (m1(None, fmt=True), True, "no well-formed answer after 3 calls"),
    (m1(None), True, "said impossible on a feasible instruction"),
    (m1([[1, 1]]), False, "invented a path on an infeasible instruction"),
    (m1([[1, 1]], ["step 3: (2, 2) -> (2, 9) is not a move to a neighbouring cell"]), True,
     "discontinuity (jump)"),
    (m1([[1, 1]], ["step 2: (0, 4) is not an allowed cell"]), True,
     "walks through a wall or forbidden cell"),
    (m1([[1, 1]], ["failed check {'type': 'ends_in', 'region': 'lab'}"]), True,
     "stops short of the goal"),
    (m1([[1, 1]], ["failed check {'type': 'never_enters', 'region': 'workshop'}"]), True,
     "walkable path violating a constraint"),
])
def test_m1_failure_types(row, feasible, expected):
    assert classify(row, GOLD, feasible)[1] == expected


def test_m2_soft_hardened():
    row = {"method": "M2", "correct": False, "formula": "G !atrium & F(lab & last)",
           "ltl_equivalent": False}
    assert classify(row, GOLD, True) == ("interpretation", "soft preference read as prohibition")
    row["formula"] = None
    assert classify(row, GOLD, True)[1] == "no syntactically valid formula after 3 calls"
