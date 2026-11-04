"""Results tables with confidence intervals and tests (plan §4.3–4.4, Day 12).

Usage:  python scripts/make_tables.py [--results results]

Reads <results>/raw/<set>/<method>.jsonl (written by scripts/run_all.py) and
writes <results>/tables/*.md plus <results>/metrics.json. Rates carry Wilson
95% intervals; paired comparisons use exact McNemar tests; cost ratios use
bootstrap intervals. Everything is reported per category.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from nlpath.benchmark import load_benchmark
from nlpath.errors import classify
from nlpath.metrics import bootstrap_mean_ci, fmt_rate, mcnemar_exact, wilson

REPO = Path(__file__).resolve().parents[1]
CATEGORIES = ["plain", "prohibition", "stop", "ordering", "soft", "infeasible"]
MAIN_METHODS = ["M0", "M1", "M2", "M3", "M2-oracle", "M3-oracle", "M3-chain"]


def load_rows(results: Path, set_name: str) -> dict[str, list[dict]]:
    raw = results / "raw" / set_name
    out = {}
    for path in sorted(raw.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if rows:
            out[path.stem] = rows
    return out


def by_category(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    return groups


def rate(rows: list[dict], key: str) -> tuple[int, int]:
    return sum(bool(r[key]) for r in rows), len(rows)


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines) + "\n"


# ----- individual tables --------------------------------------------------------


def correctness_table(rows_by_method: dict, metrics: dict) -> str:
    """Correct = all hard checks pass (feasible) or NO_PATH (infeasible)."""
    body = []
    for m in [m for m in MAIN_METHODS if m in rows_by_method]:
        groups = by_category(rows_by_method[m])
        cells = []
        for c in CATEGORIES + ["all"]:
            g = rows_by_method[m] if c == "all" else groups.get(c, [])
            k, n = rate(g, "correct")
            cells.append(fmt_rate(k, n))
            metrics.setdefault("correct", {}).setdefault(m, {})[c] = \
                {"k": k, "n": n, "rate": k / n if n else None, "ci": wilson(k, n)}
        body.append([m] + cells)
    return table(["method"] + CATEGORIES + ["all"], body)


def validity_table(rows_by_method: dict, metrics: dict) -> str:
    """Path validity on feasible instructions; NO_PATH rate on infeasible ones."""
    body = []
    for m in [m for m in MAIN_METHODS if m in rows_by_method]:
        rows = rows_by_method[m]
        feas = [r for r in rows if r["category"] != "infeasible"]
        infeas = [r for r in rows if r["category"] == "infeasible"]
        returned = [r for r in rows if r["returned_path"]]
        k_valid, n_valid = rate(feas, "valid")
        k_ret_valid = sum(r["valid"] for r in returned)
        k_detect = sum(not r["returned_path"] for r in infeas)
        body.append([m, fmt_rate(k_valid, n_valid), fmt_rate(k_ret_valid, len(returned)),
                     fmt_rate(k_detect, len(infeas))])
        metrics.setdefault("validity", {})[m] = {
            "valid_feasible": [k_valid, n_valid], "valid_when_returned": [k_ret_valid, len(returned)],
            "no_path_on_infeasible": [k_detect, len(infeas)]}
    return table(["method", "valid path (feasible instr.)", "valid among returned paths",
                  "NO_PATH on infeasible"], body)


def translation_table(rows_by_method: dict, metrics: dict) -> str:
    """M3: parsed form == gold; M2: formula equivalent to gold; failures; calls."""
    body = []
    for m, key in (("M3", "parse_exact"), ("M2", "ltl_equivalent")):
        if m not in rows_by_method:
            continue
        groups = by_category(rows_by_method[m])
        for c in CATEGORIES + ["all"]:
            g = rows_by_method[m] if c == "all" else groups.get(c, [])
            if not g:
                continue
            k, n = rate(g, key)
            failed = sum(r["calls"] == 3 and r.get("parsed", r.get("formula")) is None for r in g)
            calls = np.mean([r["calls"] for r in g])
            latency = np.mean([r["latency_s"] for r in g])
            body.append([m, c, fmt_rate(k, n), fmt_rate(failed, n), f"{calls:.2f}",
                         f"{latency:.1f}"])
            metrics.setdefault("translation", {}).setdefault(m, {})[c] = {
                "accurate": [k, n], "failed_after_retries": [failed, n],
                "mean_calls": float(calls), "mean_latency_s": float(latency)}
    return table(["method", "category", "translation accurate", "no valid output after 3 calls",
                  "mean calls", "mean latency (s)"], body)


def paraphrase_table(rows_by_method: dict, metrics: dict) -> str:
    """Do the three wordings of an instruction give the same interpretation?"""
    body = []
    for m in ("M3", "M2"):
        if m not in rows_by_method:
            continue
        per_item = defaultdict(list)
        for r in rows_by_method[m]:
            per_item[r["id"]].append(r)
        consistent = total = 0
        for rows in per_item.values():
            if len(rows) < 2:
                continue
            total += 1
            consistent += same_interpretation(m, rows)
        body.append([m, fmt_rate(consistent, total)])
        metrics.setdefault("paraphrase_consistency", {})[m] = [consistent, total]
    return table(["method", "all wordings give the same interpretation"], body)


def same_interpretation(method: str, rows: list[dict]) -> bool:
    if method == "M3":
        forms = [r["parsed"] for r in rows]
        if any(f is None for f in forms):
            return False
        norm = [(f["goal_region"], tuple(f["ordered_waypoints"]),
                 *(frozenset(f[k]) for k in ("forbidden_regions", "preferred_regions",
                                             "avoided_regions"))) for f in forms]
        return len(set(norm)) == 1
    formulas = [r["formula"] for r in rows]
    if any(f is None for f in formulas):
        return False
    from baselines.ltl_route import ltl_equivalent
    return all(ltl_equivalent(formulas[0], f) for f in formulas[1:])


def soft_table(rows_by_method: dict, metrics: dict) -> str:
    """Soft category: time in preferred / avoided regions, and detour vs M3-oracle."""
    oracle = {r["id"]: r for r in rows_by_method.get("M3-oracle", [])}
    body = []
    for m in ["M0", "M2-oracle", "M2-harden", "M2", "M3", "M3-oracle"]:
        rows = [r for r in rows_by_method.get(m, []) if r["category"] == "soft"]
        if not rows:
            continue
        pref, avoid, ratio = [], [], []
        for r in rows:
            for key, value in r["checks"].items():
                if not key.startswith("fraction_in_region:"):
                    continue
                region = key.split(":", 1)[1]
                (pref if region == "corridor" else avoid).append(value)
            o = oracle.get(r["id"])
            if r["length"] is not None and o and o["length"]:
                ratio.append(r["length"] / o["length"])
        mean = (lambda v: f"{np.mean(v):.2f}" if v else "–")
        body.append([m, str(len(rows)), mean(pref), mean(avoid), mean(ratio),
                     f"{max(ratio):.2f}" if ratio else "–"])
        metrics.setdefault("soft", {})[m] = {
            "n": len(rows), "frac_preferred": float(np.mean(pref)) if pref else None,
            "frac_avoided": float(np.mean(avoid)) if avoid else None,
            "length_ratio_vs_oracle": float(np.mean(ratio)) if ratio else None}
    return table(["method", "runs", "fraction in preferred corridor",
                  "fraction in avoided room", "length / M3-oracle length", "max ratio"], body)


def cost_table(rows_by_method: dict, metrics: dict) -> str:
    """Cost of correct paths relative to the reference optimum (gold weights)."""
    body = []
    for m in ["M0", "M1", "M2", "M3", "M2-oracle", "M3-oracle", "M3-chain"]:
        ratios = [r["cost_ratio"] for r in rows_by_method.get(m, [])
                  if r["hard_ok"] and r["cost_ratio"] is not None]
        if not ratios:
            continue
        mean, low, high = bootstrap_mean_ci(ratios)
        body.append([m, str(len(ratios)), f"{mean:.3f} [{low:.3f}–{high:.3f}]",
                     f"{max(ratios):.3f}"])
        metrics.setdefault("cost_ratio", {})[m] = {"n": len(ratios), "mean": mean,
                                                   "ci": [low, high], "max": max(ratios)}
    return table(["method", "correct paths", "cost / optimal cost (95% bootstrap CI)",
                  "max"], body)


def chain_table(rows_by_method: dict, metrics: dict) -> str:
    """Optimality gain of the product search over chaining (waypoint instructions)."""
    prod = {r["id"]: r for r in rows_by_method.get("M3-oracle", [])}
    gains = []
    for r in rows_by_method.get("M3-chain", []):
        p = prod.get(r["id"])
        if r["category"] in ("stop", "ordering") and r["cost"] and p and p["cost"]:
            gains.append(r["cost"] / p["cost"])
    if not gains:
        return ""
    mean, low, high = bootstrap_mean_ci(gains)
    worse = sum(g > 1 + 1e-9 for g in gains)
    metrics["chain_vs_product"] = {"n": len(gains), "mean_ratio": mean, "ci": [low, high],
                                   "strictly_worse": worse, "max_ratio": max(gains)}
    return table(["waypoint instructions", "chain cost / product cost", "chain strictly worse",
                  "max ratio"],
                 [[str(len(gains)), f"{mean:.3f} [{low:.3f}–{high:.3f}]",
                   fmt_rate(worse, len(gains)), f"{max(gains):.3f}"]])


def mcnemar_table(rows_by_method: dict, metrics: dict) -> str:
    """Paired exact McNemar tests on correctness (same instruction and wording)."""
    body = []
    for a, b in (("M3", "M2"), ("M3", "M1"), ("M3", "M0")):
        if a not in rows_by_method or b not in rows_by_method:
            continue
        key = (lambda r: (r["id"], r["wording"] if r["wording"] is not None else 0))
        ra = {key(r): r["correct"] for r in rows_by_method[a]}
        rb = {key(r): r["correct"] for r in rows_by_method[b]}
        for c in CATEGORIES + ["all"]:
            pairs = [k for k in ra if k in rb and
                     (c == "all" or next(r["category"] for r in rows_by_method[a]
                                         if key(r) == k) == c)]
            if not pairs:
                continue
            res = mcnemar_exact([ra[k] for k in pairs], [rb[k] for k in pairs])
            body.append([f"{a} vs {b}", c, str(res["n"]), str(res["a_only"]),
                         str(res["b_only"]), f"{res['p_value']:.2g}"])
            metrics.setdefault("mcnemar", {})[f"{a} vs {b}:{c}"] = res
    return table(["comparison", "category", "pairs", "only first correct",
                  "only second correct", "p (exact)"], body)


def nodes_table(rows_by_method: dict, metrics: dict) -> str:
    """Search effort: states expanded (M3 uses the w_min-octile heuristic; M2 has none)."""
    body = []
    for m in ["M0", "M3-oracle", "M2-oracle", "M3-chain"]:
        vals = [r["nodes_expanded"] for r in rows_by_method.get(m, [])
                if r.get("nodes_expanded") is not None]
        if vals:
            body.append([m, str(len(vals)), f"{np.mean(vals):.0f}", f"{np.median(vals):.0f}"])
            metrics.setdefault("nodes_expanded", {})[m] = {
                "mean": float(np.mean(vals)), "median": float(np.median(vals))}
    return table(["method", "runs", "mean states expanded", "median"], body)


@cache
def gold_items(set_name: str) -> dict:
    name = "benchmark.json" if set_name == "big" else "benchmark_small.json"
    return {it.id: it for it in load_benchmark(REPO / "data" / "instructions" / name)}


def failures_table(rows_by_method: dict, metrics: dict) -> str:
    """Every failure attributed to a stage and a type (plan §4.6; nlpath.errors)."""
    body = []
    for m in ["M1", "M2", "M3", "M0", "M3-oracle", "M2-oracle"]:
        rows = rows_by_method.get(m, [])
        if not rows:
            continue
        gold = gold_items(rows[0]["set"])
        counts = Counter(classify(r, gold[r["id"]].gold_intent.to_dict(),
                                  gold[r["id"]].feasible) for r in rows)
        for (stage, kind), k in sorted(counts.items(), key=lambda kv: (kv[0][0] == "ok", -kv[1])):
            body.append([m, stage, kind, str(k), f"{k / len(rows):.1%}"])
        metrics.setdefault("failures", {})[m] = {f"{s}: {t}": k for (s, t), k in counts.items()}
    return table(["method", "stage", "type", "runs", "share"], body)


# ----- driver -------------------------------------------------------------------


SECTIONS = [
    ("correctness", "Hard-constraint correctness per category", correctness_table),
    ("validity", "Path validity and infeasibility detection", validity_table),
    ("translation", "Translation accuracy (M3 intent form vs M2 LTL)", translation_table),
    ("paraphrase", "Paraphrase consistency", paraphrase_table),
    ("soft", "Soft preferences (soft category)", soft_table),
    ("cost", "Cost ratio of correct paths", cost_table),
    ("chain", "Product search vs chaining (ablation)", chain_table),
    ("mcnemar", "Paired significance tests (correctness)", mcnemar_table),
    ("nodes", "Search effort", nodes_table),
    ("failures", "Failure breakdown by stage and type", failures_table),
]


def make_tables(results: Path) -> dict:
    out_dir = results / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for set_name in ("big", "small"):
        rows_by_method = load_rows(results, set_name)
        if not rows_by_method:
            continue
        metrics: dict = {}
        doc = [f"# Results — {'40×40 benchmark' if set_name == 'big' else '15×15 maps'}\n",
               "Rates: value [Wilson 95% CI] (n). Generated by scripts/make_tables.py.\n"]
        for name, title, fn in SECTIONS:
            text = fn(rows_by_method, metrics)
            if text:
                doc.append(f"## {title}\n\n{text}")
                (out_dir / f"{set_name}_{name}.md").write_text(f"# {title}\n\n{text}")
        (out_dir / f"{set_name}_all.md").write_text("\n".join(doc))
        all_metrics[set_name] = metrics
    (results / "metrics.json").write_text(json.dumps(all_metrics, indent=1, default=float) + "\n")
    return all_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO / "results")
    args = ap.parse_args()
    make_tables(args.results)
    print(f"tables written to {args.results / 'tables'}")


if __name__ == "__main__":
    main()
