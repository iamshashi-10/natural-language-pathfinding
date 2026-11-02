"""Tests for M0, M1, the evaluator and scripts/run_all.py (Day 10)."""

import json
import sys
from pathlib import Path

import pytest

from baselines.e2e_llm import build_prompt, path_validator, run_m1
from baselines.plain_astar import run_m0
from nlpath.benchmark import Instruction, gold_ltl, instruction_checks
from nlpath.evaluate import score
from nlpath.intent import IntentForm
from nlpath.llm import FakeLLM
from nlpath.prompting import Rejected
from reference import make_map

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_all  # noqa: E402

GM = make_map(["#######",
               "#.....#",
               "#.###.#",
               "#.WWWL#",
               "#######"])


def instruction(intent, feasible=True, ref_cost=8.0):
    return Instruction("t_000", "prohibition", "template", "test", (3, 1),
                       ("a", "b", "c"), intent, gold_ltl(intent), feasible, ref_cost,
                       True, tuple(instruction_checks(intent)))


AVOID_W = instruction(IntentForm("lab", ("workshop",)))
AROUND = [(3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5)]
THROUGH = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]


# ----- M0 ---------------------------------------------------------------------


def test_m0_ignores_prohibition():
    res = run_m0(GM, "lab", (3, 1))
    assert res.path == THROUGH


# ----- evaluator ----------------------------------------------------------------


def test_score_correct_path():
    row = score(GM, AVOID_W, AROUND)
    assert row["valid"] and row["hard_ok"] and row["correct"]
    assert row["cost"] == pytest.approx(8.0) and row["cost_ratio"] == pytest.approx(1.0)
    assert row["checks"] == {"ends_in:lab": True, "never_enters:workshop": True}


def test_score_valid_but_violating_path():
    row = score(GM, AVOID_W, THROUGH)
    assert row["valid"] and not row["hard_ok"] and not row["correct"]


def test_score_wall_crossing_and_out_of_map():
    row = score(GM, AVOID_W, [(3, 1), (2, 2), (1, 3), (2, 4), (3, 5)])
    assert not row["valid"] and not row["correct"] and row["cost"] is None
    row = score(GM, AVOID_W, [(3, 1), (3, 0), (3, -1)])
    assert not row["valid"] and row["cost"] is None


def test_score_no_path():
    assert score(GM, AVOID_W, None)["correct"] is False
    infeasible = instruction(IntentForm("lab", ("workshop",)), feasible=False, ref_cost=None)
    assert score(GM, infeasible, None)["correct"] is True
    assert score(GM, infeasible, AROUND)["correct"] is False


# ----- M1 ---------------------------------------------------------------------


def test_m1_prompt_shows_map_and_start():
    prompt = build_prompt(GM, "go to the lab", (3, 1))
    assert "\n".join(GM.to_rows()) in prompt and "Start: [3, 1]" in prompt
    assert "'L' lab" in prompt


@pytest.mark.parametrize("text, expected", [
    ('{"path": [[3, 1], [3, 2]]}', [(3, 1), (3, 2)]),
    ('{"path": null}', "NO_PATH"),
])
def test_m1_validator_accepts(text, expected):
    assert path_validator(text) == expected


@pytest.mark.parametrize("text", ["nope", '{"route": []}', '{"path": []}',
                                  '{"path": [[1, 2, 3]]}', '{"path": [["a", 1]]}',
                                  '{"path": [[true, 1]]}'])
def test_m1_validator_rejects_format_errors(text):
    with pytest.raises(Rejected):
        path_validator(text)


def test_m1_walkability_is_not_fed_back():
    # A path through walls is well-formed: accepted on the first call.
    out = run_m1(FakeLLM.from_list(['{"path": [[3, 1], [2, 2], [1, 3]]}']), GM, "lab", (3, 1))
    assert out.ask.calls == 1 and out.path == [(3, 1), (2, 2), (1, 3)]


# ----- run_all ----------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "data" / "instructions" / "benchmark.json").exists(),
                    reason="benchmark not generated")
def test_run_all_dry_run(tmp_path):
    pytest.importorskip("spot")
    status = run_all.main(["--llm", "dry-run", "--limit", "6", "--out", str(tmp_path)])
    assert status == 0
    rows = {m: [json.loads(line) for line in
                (tmp_path / "raw" / "big" / f"{m}.jsonl").read_text().splitlines()]
            for m in ("M0", "M1", "M3", "M3-oracle", "M2-oracle")}
    assert len(rows["M3"]) == 18 and len(rows["M1"]) == 6 and len(rows["M0"]) == 6
    # The dry-run LLM answers with the gold intent / formula: everything correct.
    assert all(r["correct"] and r["parse_exact"] for r in rows["M3"])
    assert all(r["correct"] for r in rows["M3-oracle"] + rows["M2-oracle"])
    assert (tmp_path / "summary_big.md").exists()
