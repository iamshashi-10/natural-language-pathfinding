"""Tests for nlpath.pipeline: every returned path is verified; failures raise."""

import numpy as np
import pytest

from baselines.chaining import plan_chain
from nlpath.astar import SearchResult
from nlpath.intent import IntentError, IntentForm
from nlpath.mapgen import benchmark_maps
from nlpath.pipeline import plan
from nlpath.verify import PathVerificationError, verify_path
from reference import make_map, random_intent

ROWS = ["#######",
        "#.....#",
        "#.###.#",
        "#.WWWL#",
        "#######"]
GM = make_map(ROWS)


def test_returns_verified_path():
    res = plan(GM, IntentForm("lab", ("workshop",)), (3, 1))
    assert res.found and res.cost == pytest.approx(8.0)


def test_no_path_is_passed_through():
    res = plan(GM, IntentForm("lab", ("workshop",)), (3, 3))
    assert not res.found and "forbidden" in res.reason


def test_invalid_intent_raises():
    with pytest.raises(IntentError):
        plan(GM, IntentForm("lab", ("lab",)), (3, 1))


def test_buggy_planner_path_raises():
    # A planner that ignores the prohibition: the verifier must catch it.
    def through_workshop(problem, start):
        return SearchResult([(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)], 4.0, 0)
    with pytest.raises(PathVerificationError, match="never_enters"):
        plan(GM, IntentForm("lab", ("workshop",)), (3, 1), planner=through_workshop)


def test_buggy_planner_wall_crossing_raises():
    def through_wall(problem, start):
        return SearchResult([(3, 1), (2, 2), (1, 3)], 2.0, 0)
    with pytest.raises(PathVerificationError, match="not an allowed cell"):
        plan(GM, IntentForm("lab"), (3, 1), planner=through_wall)


def test_wrong_reported_cost_raises():
    def wrong_cost(problem, start):
        return SearchResult([(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)], 3.0, 0)
    with pytest.raises(PathVerificationError, match="cost"):
        plan(GM, IntentForm("lab"), (3, 1), planner=wrong_cost)


@pytest.mark.parametrize("planner", [None, plan_chain])
def test_all_benchmark_outputs_verify(planner):
    """Product search and chaining both pass the verifier on random intents."""
    rng = np.random.default_rng(2)
    found = 0
    for gm in benchmark_maps():
        starts = np.argwhere(gm.free & (gm.labels == -1))
        for _ in range(5):
            intent = random_intent(gm, rng)
            start = tuple(map(int, starts[rng.integers(len(starts))]))
            res = plan(gm, intent, start) if planner is None else \
                plan(gm, intent, start, planner=planner)
            if res.found:
                found += 1
                verify_path(res.path, gm, intent, start)  # explicit, for clarity
    assert found >= 50
