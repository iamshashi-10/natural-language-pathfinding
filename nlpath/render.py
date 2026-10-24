"""Matplotlib rendering of maps and paths.

Uses ``matplotlib.figure.Figure`` directly (no pyplot), so it works headless and
never opens a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from nlpath.grid import Cell, GridMap

WALL_COLOR = "#2b2b2b"
FREE_COLOR = "#ffffff"
# One fixed colour per region name, so a region looks the same on every map.
REGION_COLORS: dict[str, str] = {
    "lab": "#4e79a7",
    "workshop": "#e15759",
    "store_room": "#f28e2b",
    "print_room": "#b07aa1",
    "corridor": "#c9d9c2",
    "atrium": "#edc948",
    "office": "#76b7b2",
    "stairwell": "#9c755f",
}


def map_image(gm: GridMap) -> np.ndarray:
    """RGB image (H, W, 3) of the map: walls, free cells and region colours."""
    img = np.empty((*gm.shape, 3))
    img[:] = to_rgb(WALL_COLOR)
    img[gm.free] = to_rgb(FREE_COLOR)
    for idx, name in enumerate(gm.region_names):
        img[gm.labels == idx] = to_rgb(REGION_COLORS[name])
    return img


def draw_map(ax: Axes, gm: GridMap, path: Sequence[Cell] | None = None,
             start: Cell | None = None, goal: Cell | None = None,
             title: str | None = None, show_legend: bool = True) -> None:
    """Draw ``gm`` on ``ax``, optionally with a path and start/goal markers."""
    ax.imshow(map_image(gm), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if path:
        rows = [r for r, _ in path]
        cols = [c for _, c in path]
        ax.plot(cols, rows, color="black", linewidth=2)
    if start is not None:
        ax.plot(start[1], start[0], "o", color="#2ca02c", markersize=8,
                markeredgecolor="black", label="start")
    if goal is not None:
        ax.plot(goal[1], goal[0], "*", color="#d62728", markersize=12,
                markeredgecolor="black", label="goal")
    ax.set_title(title if title is not None else gm.map_id, fontsize=9)
    if show_legend:
        handles = [Patch(facecolor=REGION_COLORS[n], edgecolor="black", label=n)
                   for n in gm.legend()]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=7, frameon=False)


def save_map_png(gm: GridMap, out: str | Path, **kwargs) -> None:
    """Render one map (plus optional path, see ``draw_map``) to a PNG file."""
    fig = Figure(figsize=(6, 5))
    ax = fig.add_subplot()
    draw_map(ax, gm, **kwargs)
    fig.savefig(out, dpi=120, bbox_inches="tight")


def save_overview_png(maps: Sequence[GridMap], out: str | Path, ncols: int = 5) -> None:
    """Contact sheet of many maps, one panel each, with a shared legend."""
    nrows = -(-len(maps) // ncols)
    fig = Figure(figsize=(3 * ncols, 3 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, gm in zip(axes.flat, maps):
        title = f"{gm.map_id} ({gm.meta.get('tier', '')})"
        draw_map(ax, gm, title=title, show_legend=False)
    for ax in axes.flat[len(maps):]:
        ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor="black", label=n)
               for n, c in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False)
    fig.savefig(out, dpi=100, bbox_inches="tight")
