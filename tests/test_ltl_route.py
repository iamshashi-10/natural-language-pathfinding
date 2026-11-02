"""Tests for M2 (baselines/ltl_route.py), including the T6 cross-check (Day 9)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("spot")

from baselines.ltl_route import (build_automaton, harden, ltl_equivalent,  # noqa: E402
                                 ltl_validator, plan_ltl, run_m2)
from nlpath.benchmark import gold_ltl, load_benchmark  # noqa: E402
from nlpath.grid import GridMap  # noqa: E402
from nlpath.intent import IntentForm  # noqa: E402
from nlpath.llm import FakeLLM  # noqa: E402
from nlpath.pipeline import plan  # noqa: E402
from nlpath.prompting import Rejected  # noqa: E402
from nlpath.verify import is_physically_valid, path_length  # noqa: E402
from reference import make_map  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REROUTE = make_map(["#######",
                    "#.....#",
                    "#.###.#",
                    "#.WWWL#",
                    "#######"])


# ----- syntax check and validator ------------------------------------------


@pytest.mark.parametrize("answer, fragment", [
    ("not json", "not valid JSON"),
    ('{"formula": "F lab"}', '{"ltl"'),
    ('{"ltl": "G (lab"}', "not valid LTL syntax"),
    ('{"ltl": "F kitchen"}', "unknown region name(s) ['kitchen']"),
])
def test_validator_rejections(answer, fragment):
    with pytest.raises(Rejected, match=None) as err:
        ltl_validator(REROUTE.legend())(answer)
    assert fragment in str(err.value)


def test_validator_accepts_last_keyword():
    assert ltl_validator(REROUTE.legend())('{"ltl": "F(lab & last)"}') == "F(lab & last)"


def test_run_m2_retries_on_syntax_error():
    fake = FakeLLM.from_list(['{"ltl": "G !workshop & F(lab & last"}',
                              '{"ltl": "G !workshop & F(lab & last)"}'])
    out = run_m2(fake, REROUTE, "lab, not through the workshop", (3, 1))
    assert out.translation.calls == 2 and "syntax" in out.translation.errors[0]
    assert out.result.cost == pytest.approx(8.0)
    assert "#" not in fake.prompts[0]  # M2 does not see the grid either


# ----- automaton and planner ------------------------------------------------


def test_automaton_reads_labels():
    dfa = build_automaton("F(lab & last)", REROUTE.legend())
    lab, none = dfa.classes.index("lab"), len(dfa.classes) - 1
    q = dfa.delta[dfa.init, none]
    assert not dfa.accepting[q]
    assert dfa.accepting[dfa.delta[q, lab]]


def test_plan_ltl_reroute_and_last():
    # Same costs as the hand-checked product-search cases (test_product.py).
    assert plan_ltl(REROUTE, "F(lab & last)", (3, 1)).cost == pytest.approx(4.0)
    assert plan_ltl(REROUTE, "G !workshop & F(lab & last)", (3, 1)).cost == pytest.approx(8.0)
    # Start violating the formula.
    res = plan_ltl(REROUTE, "G !workshop & F(lab & last)", (3, 2))
    assert not res.found and "start" in res.reason


def test_plan_ltl_strict_ordering_infeasible():
    gm = make_map(["#########", "#..LL..S#", "#########"])
    res = plan_ltl(gm, gold_ltl(IntentForm("lab", (), ("store_room",))), (1, 1))
    assert not res.found


def test_harden():
    intent = IntentForm("lab", preferred_regions=("corridor",), avoided_regions=("workshop",))
    assert harden(intent, "F(lab & last)") == "(F(lab & last)) & G !workshop"
    # A preference alone adds no constraint.
    assert harden(IntentForm("lab", preferred_regions=("corridor",)), "F(lab & last)") \
        == "F(lab & last)"


def test_ltl_equivalence():
    assert ltl_equivalent("G !a & F(b & last)", "F(b & last) & !F a")
    assert not ltl_equivalent("F(b & last)", "F b")


# ----- T6 cross-check -------------------------------------------------------


@pytest.mark.parametrize("name", ["benchmark.json", "benchmark_small.json"])
def test_T6_m2_oracle_equals_m3_oracle_on_benchmark(name):
    """Day 9 'done when': M2-oracle cost == M3-oracle cost on all hard instructions
    (gold formula vs gold intent without soft preferences)."""
    path = REPO / "data" / "instructions" / name
    if not path.exists():
        pytest.skip("benchmark not generated")
    maps = {}
    for it in load_benchmark(path):
        gm = maps.setdefault(it.map_id, GridMap.load(REPO / "data" / "maps" / f"{it.map_id}.json"))
        m3 = plan(gm, it.gold_intent.hard_part(), it.start)
        m2 = plan_ltl(gm, it.gold_ltl, it.start)
        assert m2.found == m3.found == it.feasible, it.id
        if m2.found:
            assert m2.cost == pytest.approx(m3.cost), it.id
            assert is_physically_valid(m2.path, gm)
            assert path_length(m2.path) == pytest.approx(m2.cost)


@st.composite
def no_forbidden_problems(draw):
    """Random grids (not corner-separated) with intents without forbidden regions:
    T6 then holds on any map, since grazing only matters for forbidden cells."""
    h, w = draw(st.integers(2, 7)), draw(st.integers(2, 7))
    rows = []
    for _ in range(h):
        rows.append("".join(draw(st.sampled_from(".....#LSP")) for _ in range(w)))
    rows[0] = "L" + rows[0][1:]
    gm = make_map(rows)
    names = list(gm.legend())
    order = draw(st.permutations(names))
    m = draw(st.integers(0, len(names) - 1))
    intent = IntentForm(order[m], (), tuple(order[:m]))
    start = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    return gm, intent, start


@settings(max_examples=200)
@given(no_forbidden_problems())
def test_T6_on_random_grids(problem):
    gm, intent, start = problem
    if not gm.free[start]:
        return
    m3 = plan(gm, intent, start)
    m2 = plan_ltl(gm, gold_ltl(intent), start)
    assert m2.found == m3.found
    if m2.found:
        assert m2.cost == pytest.approx(m3.cost)
