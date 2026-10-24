"""Generate the benchmark maps, save them as JSON and render them.

Usage:  python scripts/make_maps.py [--out data/maps]

Writes, deterministically (fixed seeds in nlpath.mapgen):
    <out>/map_00.json … map_19.json        40×40 benchmark maps
    <out>/small_00.json … small_09.json    15×15 maps
    <out>/png/<map_id>.png                 one render per map
    <out>/overview.png, overview_small.png contact sheets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nlpath.mapgen import benchmark_maps, small_maps
from nlpath.render import save_map_png, save_overview_png

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "data" / "maps")
    args = parser.parse_args()

    png_dir = args.out / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    big, small = benchmark_maps(), small_maps()
    for gm in big + small:
        gm.save(args.out / f"{gm.map_id}.json")
        save_map_png(gm, png_dir / f"{gm.map_id}.png")
    save_overview_png(big, args.out / "overview.png")
    save_overview_png(small, args.out / "overview_small.png")
    print(f"wrote {len(big)} + {len(small)} maps to {args.out}")


if __name__ == "__main__":
    main()
