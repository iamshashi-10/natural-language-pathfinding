"""Stage 2: compile an intent form against a map (plan §2.4).

Pure Python, no learning:
  a) hard exclusion   — forbidden regions become untraversable (S_I);
  b) soft cost scaling — preferred 0.5, avoided 3.0, everything else 1.0;
  c) waypoint sequence — masks of (w_1, …, w_m, g) for the product search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nlpath.grid import GridMap
from nlpath.intent import IntentForm, validate

# Cell weights (plan §2.4b). All positive — required by T4.
W_PREFERRED = 0.5
W_NORMAL = 1.0
W_AVOIDED = 3.0


@dataclass(frozen=True)
class CompiledProblem:
    """Everything the planner needs; derived from the map and the intent only.

    Attributes:
        allowed: S_I = free cells minus forbidden regions (plan §3.1).
        weights: w(v) for every cell.
        sequence: masks R_{u_1}, …, R_{u_{m+1}}; the last one is the goal.
        sequence_names: the region names of ``sequence``.
    """

    allowed: np.ndarray
    weights: np.ndarray
    sequence: tuple[np.ndarray, ...]
    sequence_names: tuple[str, ...]

    @property
    def goal(self) -> np.ndarray:
        return self.sequence[-1]

    @property
    def m(self) -> int:
        """Number of waypoints."""
        return len(self.sequence) - 1


def compile_intent(gm: GridMap, intent: IntentForm) -> CompiledProblem:
    """Validate ``intent`` against ``gm``'s legend, then compile it.

    Raises ``IntentError`` if the form is not well-formed (theorem precondition).
    """
    validate(intent, gm.legend())

    allowed = gm.free.copy()
    for name in intent.forbidden_regions:
        allowed &= ~gm.region_mask(name)

    weights = np.full(gm.shape, W_NORMAL)
    for name in intent.preferred_regions:
        weights[gm.region_mask(name)] = W_PREFERRED
    for name in intent.avoided_regions:
        weights[gm.region_mask(name)] = W_AVOIDED

    sequence = tuple(gm.region_mask(name) for name in intent.sequence)
    return CompiledProblem(allowed, weights, sequence, intent.sequence)
