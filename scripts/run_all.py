"""Run every method and ablation on the benchmark — one command (plan §5, Day 10).

Usage:
    python scripts/run_all.py                    # all methods, Gemini (cached)
    python scripts/run_all.py --llm dry-run      # no network: scripted fake LLM
    python scripts/run_all.py --methods M0 M3 --sets big --limit 10

Methods (plan §4.2):
    M0          plain A* to the goal region
    M1          end-to-end LLM (sees the map, writes the path)
    M2          LLM → LTLf → automaton product planner
    M3          ours: LLM → intent form → product-graph A* → verify_path
    M3-oracle   gold intent → our planner
    M2-oracle   gold LTL → automaton planner (= M2-drop: gold LTL has no soft part)
    M2-harden   gold LTL + soft preferences made hard (soft category only)
    M3-chain    gold intent → sequential chaining (ablation)

M2 and M3 run once per wording (3 per instruction); M1 on the first wording
only (its metric is path validity, not paraphrase consistency); the others
once per instruction. About 1,300 LLM calls in total, plus retries. Every LLM answer is cached in cache/llm/, so a rerun is free and
reproducible, and an interrupted run (e.g. the daily free-tier quota) resumes
where it stopped.

Writes results/raw/<set>/<method>.jsonl (one row per run, with the path and
all metrics) and results/summary_<set>.md. A dry run writes under
results/dryrun/ instead; its numbers are NOT results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from baselines.chaining import plan_chain
from baselines.e2e_llm import run_m1
from baselines.plain_astar import run_m0
from nlpath.benchmark import Instruction, load_benchmark
from nlpath.evaluate import score
from nlpath.grid import GridMap
from nlpath.llm import LLM, FakeLLM, QuotaExhausted, default_llm
from nlpath.parser import run_m3
from nlpath.pipeline import plan

REPO = Path(__file__).resolve().parents[1]
SETS = {"big": "benchmark.json", "small": "benchmark_small.json"}
LLM_METHODS = ("M1", "M2", "M3")
ALL_METHODS = ("M0", "M1", "M2", "M3", "M3-oracle", "M2-oracle", "M2-harden", "M3-chain")
NEEDS_SPOT = ("M2", "M2-oracle", "M2-harden")


# ----- one run ----------------------------------------------------------------


def search_fields(res) -> dict:
    return {"found": res.found, "reason": res.reason, "nodes_expanded": res.nodes_expanded,
            "planner_cost": None if not res.found else res.cost}


def run_one(method: str, llm: LLM | None, gm: GridMap, it: Instruction,
            text: str | None) -> dict:
    """Run ``method`` on one instruction (and wording); return the result row."""
    row: dict = {}
    path = None
    if method == "M0":
        res = run_m0(gm, it.gold_intent.goal_region, it.start)
        row.update(search_fields(res))
        path = res.path
    elif method in ("M3-oracle", "M3-chain"):
        planner = plan_chain if method == "M3-chain" else None
        res = plan(gm, it.gold_intent, it.start) if planner is None else \
            plan(gm, it.gold_intent, it.start, planner=planner)
        row.update(search_fields(res))
        path = res.path
    elif method in ("M2-oracle", "M2-harden"):
        from baselines.ltl_route import harden, plan_ltl
        formula = it.gold_ltl if method == "M2-oracle" else harden(it.gold_intent, it.gold_ltl)
        res = plan_ltl(gm, formula, it.start)
        row.update(search_fields(res), formula=formula)
        path = res.path
    elif method == "M3":
        out = run_m3(llm, gm, text, it.start)
        parsed = out.parse.value
        row.update(calls=out.parse.calls, latency_s=round(out.parse.latency_s, 3),
                   parse_errors=out.parse.errors,
                   parsed=None if parsed is None else parsed.to_dict(),
                   parse_exact=parsed is not None and same_intent(parsed, it.gold_intent))
        if out.result is not None:
            row.update(search_fields(out.result))
            path = out.result.path
    elif method == "M2":
        from baselines.ltl_route import ltl_equivalent, run_m2
        out = run_m2(llm, gm, text, it.start)
        formula = out.translation.value
        row.update(calls=out.translation.calls,
                   latency_s=round(out.translation.latency_s, 3),
                   parse_errors=out.translation.errors, formula=formula,
                   syntax_failure=formula is None,
                   ltl_equivalent=formula is not None and ltl_equivalent(formula, it.gold_ltl))
        if out.result is not None:
            row.update(search_fields(out.result))
            path = out.result.path
    elif method == "M1":
        out = run_m1(llm, gm, text, it.start)
        row.update(calls=out.ask.calls, latency_s=round(out.ask.latency_s, 3),
                   parse_errors=out.ask.errors, said_no_path=out.ask.value == "NO_PATH",
                   format_failure=not out.ask.ok)
        path = out.path
    else:
        raise ValueError(f"unknown method {method}")
    row["path"] = None if path is None else [list(c) for c in path]
    row.update(score(gm, it, path))
    return row


def same_intent(a, b) -> bool:
    """Exact match; waypoints ordered, the other lists compared as sets."""
    return (a.goal_region == b.goal_region and a.ordered_waypoints == b.ordered_waypoints
            and all(set(getattr(a, f)) == set(getattr(b, f)) for f in
                    ("forbidden_regions", "preferred_regions", "avoided_regions")))


# ----- dry run ----------------------------------------------------------------


def dry_run_llm(items: list[Instruction], maps: dict[str, GridMap]) -> FakeLLM:
    """Scripted LLM for testing the harness without a network.

    M3/M2 prompts get the gold intent / gold formula; M1 prompts get a straight
    line from the start to the goal region that ignores walls (a caricature of a
    weak model). Only for checking that everything runs — not a result.
    """
    by_text = {text: it for it in items for text in it.texts}

    def respond(prompt: str) -> str:
        text = re.findall(r"\nInstruction: (.*)\n", prompt)[-1]
        it = by_text[text]
        if "JSON intent form" in prompt:
            return json.dumps(it.gold_intent.to_dict())
        if "LTLf" in prompt:
            return json.dumps({"ltl": it.gold_ltl})
        gm = maps[it.map_id]
        (r, c), (gr, gc) = it.start, gm.region_cells(it.gold_intent.goal_region)[0]
        path = [[r, c]]
        while (r, c) != (gr, gc):
            r += (gr > r) - (gr < r)
            c += (gc > c) - (gc < c)
            path.append([r, c])
        return json.dumps({"path": path})
    return FakeLLM(respond, model="dry-run")


# ----- driver -----------------------------------------------------------------


def summarize(rows_by_method: dict[str, list[dict]]) -> str:
    """Per method × category: runs, path validity and correctness (Day 12 adds CIs)."""
    categories = ["plain", "prohibition", "stop", "ordering", "soft", "infeasible"]
    lines = ["| method | category | runs | valid path | correct |", "|---|---|---|---|---|"]
    for method, rows in rows_by_method.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["category"]].append(row)
            groups["all"].append(row)
        for cat in [c for c in categories if c in groups] + ["all"]:
            g = groups[cat]
            valid = sum(r["valid"] for r in g) / len(g)
            correct = sum(r["correct"] for r in g) / len(g)
            lines.append(f"| {method} | {cat} | {len(g)} | {valid:.0%} | {correct:.0%} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llm", choices=["gemini", "dry-run"], default="gemini")
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--wordings", nargs="+", type=int, default=[0, 1, 2],
                    help="wordings for M2 and M3 (paraphrase consistency needs all 3)")
    ap.add_argument("--m1-wordings", nargs="+", type=int, default=[0],
                    help="wordings for M1 (one is enough: its metric is path validity)")
    ap.add_argument("--limit", type=int, default=None, help="first N instructions per set")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or REPO / ("results/dryrun" if args.llm == "dry-run" else "results")
    maps = {p.stem: GridMap.load(p) for p in (REPO / "data" / "maps").glob("*.json")}
    benchmarks = {s: load_benchmark(REPO / "data" / "instructions" / SETS[s])[:args.limit]
                  for s in args.sets}
    if any(m in NEEDS_SPOT for m in args.methods):
        try:
            import spot  # noqa: F401
        except ImportError:
            print("Spot is not installed; skipping", NEEDS_SPOT, file=sys.stderr)
            args.methods = [m for m in args.methods if m not in NEEDS_SPOT]
    if args.llm == "dry-run":
        llm = dry_run_llm([it for b in benchmarks.values() for it in b], maps)
    else:
        llm = default_llm() if any(m in LLM_METHODS for m in args.methods) else None

    status = 0
    for set_name, items in benchmarks.items():
        raw_dir = out / "raw" / set_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows_by_method = {}
        for method in args.methods:
            started, rows = time.monotonic(), []
            try:
                for it in items:
                    if method == "M2-harden" and it.category != "soft":
                        continue
                    gm = maps[it.map_id]
                    wordings = (args.m1_wordings if method == "M1" else args.wordings) \
                        if method in LLM_METHODS else [None]
                    for w in wordings:
                        row = {"method": method, "set": set_name, "id": it.id,
                               "category": it.category, "source": it.source,
                               "map_id": it.map_id, "wording": w}
                        row.update(run_one(method, llm, gm, it,
                                           None if w is None else it.texts[w]))
                        rows.append(row)
            except QuotaExhausted:
                print(f"{method}: LLM quota exhausted after {len(rows)} runs; partial "
                      "results saved. Rerun later — cached answers are reused.",
                      file=sys.stderr)
                status = 2
            with open(raw_dir / f"{method}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            rows_by_method[method] = rows
            print(f"[{set_name}] {method}: {len(rows)} runs in "
                  f"{time.monotonic() - started:.1f}s")
        summary = summarize({m: r for m, r in rows_by_method.items() if r})
        (out / f"summary_{set_name}.md").write_text(summary)
        print(summary)
    return status


if __name__ == "__main__":
    sys.exit(main())
