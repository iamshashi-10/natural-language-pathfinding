"""M0: plain A* from the start to the goal region, ignoring everything else.

The floor of the comparison (plan §4.2). It is told the goal region (from the
gold intent) and nothing else: no prohibitions, waypoints or preferences.
"""

from __future__ import annotations

import numpy as np

from nlpath.astar import SearchResult, plan_grid
from nlpath.grid import Cell, GridMap


def run_m0(gm: GridMap, goal_region: str, start: Cell) -> SearchResult:
    """Shortest physically valid path from ``start`` into ``goal_region``."""
    return plan_grid(gm.free, np.ones(gm.shape), start, gm.region_mask(goal_region))
