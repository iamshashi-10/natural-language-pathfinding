"""Grid map data structures and JSON load/save.

A map is a 2-D grid of cells addressed as ``(row, col)``. Each cell is either a
wall or free; each free cell carries at most one region label (plan §3.1:
labelling λ : F → R ∪ {⊥}). Regions are identified by name, and in this project
each name occurs at most once per map, so "the lab" is never ambiguous.

On disk a map is JSON with the grid stored as text rows, one character per cell:

    '#'  wall
    '.'  free, unlabelled (⊥)
    'L'  free, labelled with the region whose letter is 'L' (see ``legend``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Cell = tuple[int, int]

# Region vocabulary (plan §2.6) and the fixed letter used for each on disk.
REGION_LETTERS: dict[str, str] = {
    "lab": "L",
    "workshop": "W",
    "store_room": "S",
    "print_room": "P",
    "corridor": "C",
    "atrium": "A",
    "office": "O",
    "stairwell": "T",
}
REGION_VOCAB: tuple[str, ...] = tuple(REGION_LETTERS)

WALL_CHAR = "#"
FREE_CHAR = "."
NO_LABEL = -1


@dataclass(frozen=True)
class GridMap:
    """A grid map with labelled regions.

    Attributes:
        map_id: identifier, e.g. ``"map_07"``.
        free: bool array of shape (H, W); ``False`` means wall.
        labels: int array of shape (H, W); index into ``region_names``,
            or ``NO_LABEL`` (-1) for unlabelled cells and walls.
        region_names: names of the regions present on this map.
        meta: generator provenance (seed, tier, parameters).
    """

    map_id: str
    free: np.ndarray
    labels: np.ndarray
    region_names: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.free.ndim != 2 or self.free.dtype != bool:
            raise ValueError("free must be a 2-D bool array")
        if self.labels.shape != self.free.shape:
            raise ValueError("labels and free must have the same shape")
        if len(set(self.region_names)) != len(self.region_names):
            raise ValueError(f"duplicate region names: {self.region_names}")
        unknown = set(self.region_names) - set(REGION_VOCAB)
        if unknown:
            raise ValueError(f"region names not in vocabulary: {sorted(unknown)}")
        n = len(self.region_names)
        if self.labels.min() < NO_LABEL or self.labels.max() >= n:
            raise ValueError("labels contain an index outside region_names")
        # λ is defined only on free cells (plan §3.1).
        if np.any((self.labels != NO_LABEL) & ~self.free):
            raise ValueError("a wall cell carries a region label")

    # ----- queries -------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        h, w = self.shape
        return 0 <= r < h and 0 <= c < w

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and bool(self.free[cell])

    def label_at(self, cell: Cell) -> str | None:
        """Region name of ``cell``, or ``None`` if unlabelled or a wall."""
        idx = int(self.labels[cell])
        return None if idx == NO_LABEL else self.region_names[idx]

    def region_mask(self, name: str) -> np.ndarray:
        """Bool mask of the cells of region ``name`` (R_name in plan §3.1)."""
        if name not in self.region_names:
            raise KeyError(f"region {name!r} not on map {self.map_id}")
        return self.labels == self.region_names.index(name)

    def region_cells(self, name: str) -> list[Cell]:
        rows, cols = np.nonzero(self.region_mask(name))
        return [(int(r), int(c)) for r, c in zip(rows, cols)]

    def legend(self) -> list[str]:
        """Region names, in a fixed order — the only map information the parser sees."""
        return sorted(self.region_names, key=REGION_VOCAB.index)

    # ----- serialisation -------------------------------------------------------

    def to_rows(self) -> list[str]:
        """Encode the grid as text rows (see module docstring)."""
        letters = [REGION_LETTERS[name] for name in self.region_names]
        rows = []
        for r in range(self.shape[0]):
            chars = []
            for c in range(self.shape[1]):
                if not self.free[r, c]:
                    chars.append(WALL_CHAR)
                elif self.labels[r, c] == NO_LABEL:
                    chars.append(FREE_CHAR)
                else:
                    chars.append(letters[self.labels[r, c]])
            rows.append("".join(chars))
        return rows

    def to_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "height": self.shape[0],
            "width": self.shape[1],
            "legend": {REGION_LETTERS[n]: n for n in self.legend()},
            "rows": self.to_rows(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GridMap:
        rows: list[str] = data["rows"]
        h, w = data["height"], data["width"]
        if len(rows) != h or any(len(row) != w for row in rows):
            raise ValueError("rows do not match height/width")
        letter_to_name: dict[str, str] = data["legend"]
        for letter, name in letter_to_name.items():
            if REGION_LETTERS.get(name) != letter:
                raise ValueError(f"legend letter {letter!r} does not match {name!r}")
        region_names = tuple(sorted(letter_to_name.values(), key=REGION_VOCAB.index))
        letter_to_idx = {REGION_LETTERS[n]: i for i, n in enumerate(region_names)}

        free = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), NO_LABEL, dtype=np.int8)
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == WALL_CHAR:
                    continue
                free[r, c] = True
                if ch != FREE_CHAR:
                    if ch not in letter_to_idx:
                        raise ValueError(f"unknown cell character {ch!r} at {(r, c)}")
                    labels[r, c] = letter_to_idx[ch]
        return cls(data["map_id"], free, labels, region_names, data.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> GridMap:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridMap):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]  # mutable numpy fields
