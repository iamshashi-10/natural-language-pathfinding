"""Test helpers: maps from text rows, random intents, and the reference solver.

The reference solver itself lives in ``nlpath.reference`` (it also labels the
benchmark); it is re-exported here for the tests.
"""

from __future__ import annotations

import numpy as np

from nlpath.grid import REGION_LETTERS, GridMap
from nlpath.intent import IntentForm, validate
from nlpath.reference import reference_solve  # noqa: F401  (re-export)


def make_map(rows: list[str], map_id: str = "test") -> GridMap:
    """GridMap from text rows; the legend is inferred from the letters used."""
    letters = {ch for row in rows for ch in row} - {"#", "."}
    by_letter = {v: k for k, v in REGION_LETTERS.items()}
    return GridMap.from_dict({"map_id": map_id, "height": len(rows),
                              "width": len(rows[0]), "rows": rows,
                              "legend": {ch: by_letter[ch] for ch in letters}})


def random_intent(gm, rng):
    """A random well-formed intent for ``gm`` (goal, ≤2 waypoints, 1 forbidden, soft)."""
    names = list(gm.region_names)
    rng.shuffle(names)
    m = int(rng.integers(0, 3))
    seq, rest = names[:m + 1], names[m + 1:]
    forbidden = tuple(rest[:1]) if rest and rng.random() < 0.5 else ()
    soft = [n for n in names if n not in forbidden]
    preferred = (soft[0],) if rng.random() < 0.5 else ()
    avoided = (soft[1],) if rng.random() < 0.5 and soft[1:] else ()
    intent = IntentForm(seq[-1], forbidden, tuple(seq[:-1]), preferred, avoided)
    validate(intent, gm.legend())
    return intent
