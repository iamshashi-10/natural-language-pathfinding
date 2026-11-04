"""Figures for the report (plan §5, Day 14).

Usage:  python scripts/make_figures.py [--results results]

Reads <results>/raw/big/*.jsonl and writes <results>/figures/:
    validity.png        valid paths and correct answers per method (Wilson 95% CI)
    m1_vs_m3.png        the same instruction: M1's path vs ours
    soft_harden.png     a soft instruction: M2-drop, M2-harden and M3 side by side
    t5_detour.png       soft preferences: path length / shortest hard path, vs the T5 bound
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from nlpath.benchmark import load_benchmark
from nlpath.grid import GridMap
from nlpath.metrics import wilson
from nlpath.render import draw_map

REPO = Path(__file__).resolve().parents[1]
# Reference categorical palette (dataviz skill), fixed order; text in neutral ink.
SERIES = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHODS = ["M0", "M1", "M2", "M3", "M2-oracle", "M3-oracle"]


def load(results: Path) -> dict[str, list[dict]]:
    raw = results / "raw" / "big"
    return {p.stem: [json.loads(x) for x in p.read_text().splitlines() if x]
            for p in raw.glob("*.jsonl")}


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def validity_figure(rows: dict, out: Path) -> None:
    """Horizontal grouped bars: valid path (feasible instr.) and correct (all)."""
    methods = [m for m in METHODS if m in rows]
    measures = [("valid path (walkable, reaches the goal)",
                 lambda r: r["category"] != "infeasible", "valid"),
                ("correct (hard constraints / NO_PATH)", lambda r: True, "correct")]
    fig = Figure(figsize=(7, 0.55 * len(methods) + 1.4))
    ax = fig.add_subplot()
    style(ax)
    y = np.arange(len(methods))
    h = 0.36
    for j, (label, keep, key) in enumerate(measures):
        vals, lo, hi = [], [], []
        for m in methods:
            g = [r for r in rows[m] if keep(r)]
            k, n = sum(r[key] for r in g), len(g)
            low, high = wilson(k, n)
            vals.append(k / n)
            lo.append(max(0.0, k / n - low))   # clamp float noise at 0% / 100%
            hi.append(max(0.0, high - k / n))
        pos = y + (j - 0.5) * (h + 0.04)
        ax.barh(pos, vals, height=h, color=SERIES[j], label=label,
                xerr=[lo, hi], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for p, v, up in zip(pos, vals, hi):
            # Label just past the error bar, so text never sits on a mark.
            ax.text(v + up + 0.015, p, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.set_title("Path validity and correctness on the 40×40 benchmark", color=INK,
                 fontsize=10, loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")


def _maps() -> dict[str, GridMap]:
    return {p.stem: GridMap.load(p) for p in (REPO / "data" / "maps").glob("*.json")}


def panels(out: Path, items: list[tuple[GridMap, list | None, tuple, str]],
           caption: str = "") -> None:
    fig = Figure(figsize=(4.2 * len(items), 4.0))
    axes = fig.subplots(1, len(items))
    for i, (ax, (gm, path, start, title)) in enumerate(zip(axes, items)):
        draw_map(ax, gm, path=path, start=start, goal=path[-1] if path else None,
                 title=title, show_legend=i == len(items) - 1)
    if caption:
        fig.suptitle(caption, fontsize=10, color=INK, y=1.0)
    fig.savefig(out, dpi=130, bbox_inches="tight")


def m1_vs_m3_figure(rows: dict, out: Path) -> bool:
    """First instruction where M1 is invalid and M3 (wording 0) is correct."""
    m3 = {r["id"]: r for r in rows.get("M3", []) if r["wording"] == 0}
    gold = {it.id: it for it in load_benchmark(REPO / "data/instructions/benchmark.json")}
    maps = _maps()
    for r in rows.get("M1", []):
        o = m3.get(r["id"])
        if r["path"] and not r["valid"] and o and o["correct"] and r["category"] == "prohibition":
            it = gold[r["id"]]
            gm = maps[it.map_id]
            path = [tuple(c) for c in r["path"] if gm.in_bounds(tuple(c))]
            panels(out, [(gm, path, it.start, "M1: the LLM writes the path"),
                         (gm, [tuple(c) for c in o["path"]], it.start,
                          "M3 (ours): intent form → A*")],
                   caption=f"“{it.texts[0]}”")
            return True
    return False


def soft_harden_figure(rows: dict, out: Path) -> bool:
    """A soft 'avoid' instruction where hardening forces the longest detour."""
    by = {m: {r["id"]: r for r in rows.get(m, [])} for m in ("M2-oracle", "M2-harden", "M3-oracle")}
    gold = {it.id: it for it in load_benchmark(REPO / "data/instructions/benchmark.json")}
    best, best_ratio = None, 1.0
    for i, h in by["M2-harden"].items():
        d = by["M2-oracle"].get(i)
        if h["length"] and d and d["length"] and h["length"] / d["length"] > best_ratio:
            best, best_ratio = i, h["length"] / d["length"]
    if best is None:
        return False
    it, maps = gold[best], _maps()
    gm = maps[it.map_id]
    panels(out, [(gm, [tuple(c) for c in by[m][best]["path"]], it.start, title) for m, title in
                 (("M2-oracle", "M2-drop: preference ignored"),
                  ("M2-harden", "M2-harden: “avoid” made a prohibition"),
                  ("M3-oracle", "M3: weighted trade-off"))],
           caption=f"“{it.texts[0]}”")
    return True


def t5_figure(rows: dict, out: Path) -> bool:
    """Length of our soft-preference path / shortest path satisfying the hard part.

    M2-oracle plans the gold formula (hard part only) with unit costs, so its
    length is L* (theorem T6); T5 bounds the ratio by w_max / w_min.
    """
    shortest = {r["id"]: r["length"] for r in rows.get("M2-oracle", []) if r["length"]}
    ratios = [r["length"] / shortest[r["id"]] for r in rows.get("M3-oracle", [])
              if r["category"] == "soft" and r["length"] and r["id"] in shortest]
    if not ratios:
        return False
    fig = Figure(figsize=(6.5, 3.2))
    ax = fig.add_subplot()
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.hist(ratios, bins=np.arange(1.0, max(ratios) + 0.1, 0.05), color=SERIES[0],
            edgecolor="white", linewidth=2)
    for bound, label in ((2.0, "T5 bound, preferred only (1/0.5)"),
                         (3.0, "avoided only (3/1)")):
        ax.axvline(bound, color=MUTED, linestyle="--", linewidth=1)
        ax.text(bound, ax.get_ylim()[1] * 0.92, " " + label, fontsize=7, color=MUTED)
    ax.set_xlim(0.98, 3.4)
    ax.set_xlabel("path length / shortest length satisfying the hard constraints", color=INK)
    ax.set_ylabel("soft instructions", color=INK)
    ax.set_title(f"Soft preferences cost little detour (max {max(ratios):.2f}×; "
                 "bound 6×)", color=INK, fontsize=10, loc="left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return True


def make_figures(results: Path) -> list[str]:
    rows = load(results)
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if rows:
        validity_figure(rows, out / "validity.png")
        made.append("validity.png")
    for name, fn in (("m1_vs_m3.png", m1_vs_m3_figure), ("soft_harden.png", soft_harden_figure),
                     ("t5_detour.png", t5_figure)):
        if fn(rows, out / name):
            made.append(name)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=REPO / "results")
    args = ap.parse_args()
    print("figures:", ", ".join(make_figures(args.results)))


if __name__ == "__main__":
    main()
