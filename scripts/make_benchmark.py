"""Generate the instruction benchmark and print its statistics.

Usage:  python scripts/make_benchmark.py

Writes (deterministically; needs the maps from scripts/make_maps.py):
    data/instructions/benchmark.json        165 instructions × 3 wordings, 40×40 maps
    data/instructions/benchmark_small.json  20 instructions × 3 wordings, 15×15 maps
    data/instructions/stats.json            counts per category / source / label
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nlpath.benchmark import SMALL_PLAN, generate, save_benchmark
from nlpath.grid import GridMap

REPO = Path(__file__).resolve().parents[1]
MAPS = REPO / "data" / "maps"
OUT = REPO / "data" / "instructions"


def stats(items) -> dict:
    return {
        "instructions": len(items),
        "inputs": sum(len(it.texts) for it in items),
        "per_category": dict(Counter(it.category for it in items)),
        "per_source": dict(Counter(it.source for it in items)),
        "feasible": sum(it.feasible for it in items),
        "binding": sum(it.binding for it in items),
        "maps_used": len({it.map_id for it in items}),
    }


def main() -> None:
    big = [GridMap.load(MAPS / f"map_{i:02d}.json") for i in range(20)]
    small = [GridMap.load(MAPS / f"small_{i:02d}.json") for i in range(10)]
    OUT.mkdir(parents=True, exist_ok=True)

    items = generate(big)
    small_items = generate(small, template_plan=SMALL_PLAN, handwritten=[], seed=3001)
    save_benchmark(items, OUT / "benchmark.json")
    save_benchmark(small_items, OUT / "benchmark_small.json")
    summary = {"benchmark": stats(items), "benchmark_small": stats(small_items)}
    (OUT / "stats.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
