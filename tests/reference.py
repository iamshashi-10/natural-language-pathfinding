"""Reference solver: explicit product graph in networkx (plan §3.5, §4.1).

Written directly from the definitions in plan §3.1–3.2 and deliberately
independent of nlpath.astar / nlpath.product, so agreement between the two is
evidence of correctness. Also the source of benchmark ``feasible`` labels.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from nlpath.grid import REGION_LETTERS, GridMap
from nlpath.intent import IntentForm

W = {"preferred": 0.5, "normal": 1.0, "avoided": 3.0}


def make_map(rows: list[str], map_id: str = "test") -> GridMap:
    """GridMap from text rows; the legend is inferred from the letters used."""
    letters = {ch for row in rows for ch in row} - {"#", "."}
    by_letter = {v: k for k, v in REGION_LETTERS.items()}
    return GridMap.from_dict({"map_id": map_id, "height": len(rows),
                              "width": len(rows[0]), "rows": rows,
                              "legend": {ch: by_letter[ch] for ch in letters}})


def reference_solve(gm: GridMap, intent: IntentForm,
                    start: tuple[int, int]) -> float:
    """Optimal cost J of an I-satisfying path from ``start``; inf if none exists.

    Assumes ``intent`` is well-formed. Returns inf for an inadmissible start.
    """
    h, w = gm.shape
    seq = [gm.labels == gm.region_names.index(n) for n in intent.sequence]
    final = len(seq)
    allowed = gm.free.copy()
    for n in intent.forbidden_regions:
        allowed &= gm.labels != gm.region_names.index(n)
    weight = np.full(gm.shape, W["normal"])
    for n in intent.preferred_regions:
        weight[gm.labels == gm.region_names.index(n)] = W["preferred"]
    for n in intent.avoided_regions:
        weight[gm.labels == gm.region_names.index(n)] = W["avoided"]

    def element_of(cell) -> int | None:
        for j, mask in enumerate(seq):
            if mask[cell]:
                return j
        return None

    if not allowed[start]:
        return math.inf
    j0 = element_of(start)
    if j0 is not None and j0 >= 1:
        return math.inf
    k0 = 1 if j0 == 0 else 0

    g = nx.DiGraph()
    for r in range(h):
        for c in range(w):
            if not allowed[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    u = (r + dr, c + dc)
                    if (dr, dc) == (0, 0) or not (0 <= u[0] < h and 0 <= u[1] < w):
                        continue
                    if not allowed[u]:
                        continue
                    if dr and dc and not (allowed[r, c + dc] and allowed[r + dr, c]):
                        continue
                    j = element_of(u)
                    for k in range(final + 1):
                        # Strict ordering: may enter element j only if j <= k.
                        if j is not None and j > k:
                            continue
                        k2 = k + 1 if j == k else k
                        g.add_edge(((r, c), k), (u, k2),
                                   weight=math.hypot(dr, dc) * weight[u])
    g.add_node((start, k0))
    dist = nx.single_source_dijkstra_path_length(g, (start, k0))
    goals = [d for (v, k), d in dist.items() if k == final and seq[-1][v]]
    return min(goals, default=math.inf)
