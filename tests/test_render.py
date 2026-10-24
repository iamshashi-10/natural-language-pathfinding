"""Smoke tests for nlpath.render."""

import numpy as np

from nlpath.mapgen import SMALL, generate_map
from nlpath.render import REGION_COLORS, map_image, save_map_png, save_overview_png
from nlpath.grid import REGION_VOCAB


def test_every_region_has_a_colour():
    assert set(REGION_COLORS) == set(REGION_VOCAB)


def test_map_image_shape_and_walls():
    gm = generate_map("m", 0, SMALL)
    img = map_image(gm)
    assert img.shape == (15, 15, 3)
    assert np.allclose(img[0, 0], img[-1, -1])  # both border walls


def test_png_files_written(tmp_path):
    gm = generate_map("m", 0, SMALL)
    free = [tuple(map(int, x)) for x in np.argwhere(gm.free)]
    save_map_png(gm, tmp_path / "m.png", path=free[:3], start=free[0], goal=free[2])
    save_overview_png([gm, gm], tmp_path / "o.png", ncols=2)
    for f in ("m.png", "o.png"):
        data = (tmp_path / f).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
