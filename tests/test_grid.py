"""Tests for nlpath.grid: validation, queries and JSON round-trip."""

import numpy as np
import pytest

from nlpath.grid import NO_LABEL, GridMap

ROWS = [
    "######",
    "#LL.C#",
    "#LL#C#",
    "#..#.#",
    "######",
]


def tiny_map() -> GridMap:
    data = {"map_id": "tiny", "height": 5, "width": 6,
            "legend": {"L": "lab", "C": "corridor"}, "rows": ROWS, "meta": {"seed": 0}}
    return GridMap.from_dict(data)


def test_rows_round_trip():
    gm = tiny_map()
    assert gm.to_rows() == ROWS
    assert GridMap.from_dict(gm.to_dict()) == gm


def test_save_load(tmp_path):
    gm = tiny_map()
    gm.save(tmp_path / "m.json")
    assert GridMap.load(tmp_path / "m.json") == gm


def test_queries():
    gm = tiny_map()
    assert gm.shape == (5, 6)
    assert gm.legend() == ["lab", "corridor"]
    assert gm.label_at((1, 1)) == "lab"
    assert gm.label_at((1, 3)) is None      # free, unlabelled
    assert gm.label_at((0, 0)) is None      # wall
    assert gm.is_free((3, 1)) and not gm.is_free((2, 3)) and not gm.is_free((-1, 0))
    assert gm.region_cells("corridor") == [(1, 4), (2, 4)]
    assert gm.region_mask("lab").sum() == 4
    with pytest.raises(KeyError):
        gm.region_mask("workshop")


def make(free, labels, names):
    return GridMap("m", np.array(free, dtype=bool), np.array(labels, dtype=np.int8),
                   tuple(names))


def test_rejects_label_on_wall():
    with pytest.raises(ValueError, match="wall"):
        make([[False]], [[0]], ["lab"])


def test_rejects_unknown_name():
    with pytest.raises(ValueError, match="vocabulary"):
        make([[True]], [[0]], ["kitchen"])


def test_rejects_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        make([[True, True]], [[0, 1]], ["lab", "lab"])


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        make([[True, True]], [[NO_LABEL]], [])


def test_rejects_label_index_out_of_range():
    with pytest.raises(ValueError, match="index"):
        make([[True]], [[1]], ["lab"])


def test_from_dict_rejects_bad_legend_and_chars():
    good = tiny_map().to_dict()
    with pytest.raises(ValueError, match="legend"):
        GridMap.from_dict({**good, "legend": {"X": "lab", "C": "corridor"}})
    with pytest.raises(ValueError, match="character"):
        GridMap.from_dict({**good, "rows": [ROWS[0], "#LLWC#", *ROWS[2:]]})
    with pytest.raises(ValueError, match="height"):
        GridMap.from_dict({**good, "rows": ROWS[:-1]})
