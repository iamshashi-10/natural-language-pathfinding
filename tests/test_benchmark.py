"""Tests for the instruction benchmark (Day 7): format, gold labels, coverage."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from nlpath.benchmark import gold_ltl, instruction_checks, intent_from_slots, load_benchmark
from nlpath.grid import NO_LABEL, GridMap
from nlpath.intent import IntentForm, validate
from nlpath.pipeline import plan
from nlpath.reference import reference_solve

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "instructions" / "benchmark.json"
SMALL = REPO / "data" / "instructions" / "benchmark_small.json"

pytestmark = pytest.mark.skipif(not BENCH.exists(), reason="benchmark not generated")


@pytest.fixture(scope="module")
def items():
    return load_benchmark(BENCH) + load_benchmark(SMALL)


@pytest.fixture(scope="module")
def maps():
    return {p.stem: GridMap.load(p) for p in (REPO / "data" / "maps").glob("*.json")}


def test_gold_ltl_matches_plan_example():
    # Plan §3.3 example: "visit A before B, then the lab, avoid the workshop"
    # (A, B stand for store_room and print_room).
    intent = IntentForm("lab", ("workshop",), ("store_room", "print_room"))
    assert gold_ltl(intent) == ("G !workshop & (!(print_room | lab) U "
                                "(store_room & (!lab U (print_room & F(lab & last)))))")
    assert gold_ltl(IntentForm("lab", ("workshop", "office"))) == \
        "G !(workshop | office) & F(lab & last)"


def test_soft_preferences_become_fraction_checks():
    intent = intent_from_slots("G C Y", {"G": "lab", "C": "corridor", "Y": "office"})
    assert intent.preferred_regions == ("corridor",) and intent.avoided_regions == ("office",)
    assert instruction_checks(intent)[-2:] == [
        {"type": "fraction_in_region", "region": "corridor"},
        {"type": "fraction_in_region", "region": "office"}]


def test_counts_match_plan():
    big = load_benchmark(BENCH)
    assert len(big) == 165 and sum(len(it.texts) for it in big) == 495
    assert Counter(it.category for it in big) == {
        "plain": 20, "prohibition": 35, "stop": 30, "ordering": 30, "soft": 35,
        "infeasible": 15}
    assert Counter(it.source for it in big) == {"template": 135, "handwritten": 30}
    assert len({it.id for it in big}) == 165


def test_instructions_are_well_formed(items, maps):
    for it in items:
        gm = maps[it.map_id]
        validate(it.gold_intent, gm.legend())
        assert gm.free[it.start] and gm.labels[it.start] == NO_LABEL  # outside every region
        assert len(set(it.texts)) == 3
        assert all("{" not in t and "_" not in t for t in it.texts)
        assert it.gold_ltl == gold_ltl(it.gold_intent)
        assert list(it.checks) == instruction_checks(it.gold_intent)
        assert it.feasible == (it.category != "infeasible")


def test_labels_match_reference_and_planner(items, maps):
    """Feasibility and cost labels agree with the reference solver, and the planner
    (fed the gold intent, i.e. M3-oracle) agrees with both (T3, T4 on the benchmark)."""
    for it in items:
        gm = maps[it.map_id]
        ref = reference_solve(gm, it.gold_intent, it.start)
        assert it.feasible == (not math.isinf(ref)), it.id
        res = plan(gm, it.gold_intent, it.start)
        assert res.found == it.feasible, it.id
        if it.feasible:
            assert it.ref_cost == pytest.approx(ref) == pytest.approx(res.cost), it.id


def test_prohibitions_mostly_change_the_route():
    big = load_benchmark(BENCH)
    prohib = [it for it in big if it.category == "prohibition"]
    assert np.mean([it.binding for it in prohib]) >= 0.8


def test_gold_ltl_parses_in_spot(items):
    pytest.importorskip("spot")
    from baselines.ltl_route import parse_ltl
    for it in items:
        parse_ltl(it.gold_ltl, list(it.gold_intent.sequence) + list(it.gold_intent.forbidden_regions))
