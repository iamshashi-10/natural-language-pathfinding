"""Tests for nlpath.intent: schema parsing and well-formedness."""

import pytest

from nlpath.intent import IntentError, IntentForm, validate

LEGEND = ["lab", "workshop", "store_room", "corridor"]


def test_round_trip_and_sequence():
    d = {"goal_region": "lab", "forbidden_regions": ["workshop"],
         "ordered_waypoints": ["store_room"], "preferred_regions": ["corridor"],
         "avoided_regions": []}
    intent = IntentForm.from_dict(d)
    assert intent.to_dict() == d
    assert intent.sequence == ("store_room", "lab")
    assert intent.hard_part() == IntentForm("lab", ("workshop",), ("store_room",))


def test_missing_lists_default_to_empty():
    assert IntentForm.from_dict({"goal_region": "lab"}) == IntentForm("lab")


@pytest.mark.parametrize("data, match", [
    ([], "JSON object"),
    ({}, "goal_region"),
    ({"goal_region": ""}, "goal_region"),
    ({"goal_region": 3}, "goal_region"),
    ({"goal_region": "lab", "forbidden_regions": "workshop"}, "forbidden_regions"),
    ({"goal_region": "lab", "avoided_regions": [1]}, "avoided_regions"),
    ({"goal_region": "lab", "path": [[0, 0]]}, "unknown field"),
])
def test_schema_errors(data, match):
    with pytest.raises(IntentError, match=match):
        IntentForm.from_dict(data)


def test_well_formed_passes():
    validate(IntentForm("lab", ("workshop",), ("store_room",), ("corridor",)), LEGEND)


@pytest.mark.parametrize("intent, match", [
    (IntentForm("kitchen"), "unknown region 'kitchen'"),
    (IntentForm("lab", forbidden_regions=("garage",)), "forbidden_regions"),
    (IntentForm("lab", ordered_waypoints=("lab",)), "more than once"),
    (IntentForm("lab", ordered_waypoints=("store_room", "store_room")), "more than once"),
    (IntentForm("lab", ("lab",)), "also forbidden"),
    (IntentForm("lab", ("store_room",), ("store_room",)), "also forbidden"),
    (IntentForm("lab", preferred_regions=("corridor",), avoided_regions=("corridor",)),
     "both preferred and avoided"),
])
def test_not_well_formed(intent, match):
    with pytest.raises(IntentError, match=match):
        validate(intent, LEGEND)


def test_soft_preferences_may_overlap_sequence():
    # Avoiding the goal region is odd but well-formed: it only changes weights.
    validate(IntentForm("lab", avoided_regions=("lab",)), LEGEND)
