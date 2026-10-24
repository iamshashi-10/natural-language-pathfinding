"""Tests for nlpath.compiler on a hand-written map."""

import numpy as np
import pytest

from nlpath.compiler import W_AVOIDED, W_NORMAL, W_PREFERRED, compile_intent
from nlpath.intent import IntentError, IntentForm
from reference import make_map

ROWS = [
    "#######",
    "#CCCCC#",
    "#.WW.L#",
    "#S.#..#",
    "#######",
]


def test_exclusion_weights_and_sequence():
    gm = make_map(ROWS)
    intent = IntentForm("lab", forbidden_regions=("workshop",),
                        ordered_waypoints=("store_room",),
                        preferred_regions=("corridor",))
    p = compile_intent(gm, intent)

    # a) forbidden cells are removed from the free cells, nothing else is.
    assert not p.allowed[2, 2] and not p.allowed[2, 3]
    assert (p.allowed == (gm.free & ~gm.region_mask("workshop"))).all()
    # b) weights.
    assert (p.weights[1, 1:6] == W_PREFERRED).all()
    assert p.weights[3, 1] == W_NORMAL and p.weights[2, 2] == W_NORMAL
    # c) sequence = (waypoints…, goal), goal last.
    assert p.sequence_names == ("store_room", "lab")
    assert p.m == 1
    assert p.sequence[0][3, 1] and p.goal[2, 5] and p.goal.sum() == 1


def test_avoided_weight():
    gm = make_map(ROWS)
    p = compile_intent(gm, IntentForm("lab", avoided_regions=("workshop",)))
    workshop = gm.region_mask("workshop")
    assert (p.weights[workshop] == W_AVOIDED).all()
    assert (p.weights[~workshop] == W_NORMAL).all()
    assert p.allowed[2, 2]  # avoided is soft: still allowed


def test_all_weights_positive():
    gm = make_map(ROWS)
    p = compile_intent(gm, IntentForm("lab", (), (), ("corridor",), ("workshop",)))
    assert (p.weights > 0).all()


def test_compile_validates_first():
    with pytest.raises(IntentError):
        compile_intent(make_map(ROWS), IntentForm("lab", ("lab",)))
    with pytest.raises(IntentError):
        compile_intent(make_map(ROWS), IntentForm("office"))  # not on this map


def test_compile_does_not_mutate_map():
    gm = make_map(ROWS)
    before = gm.free.copy()
    compile_intent(gm, IntentForm("lab", ("workshop",)))
    assert np.array_equal(gm.free, before)
