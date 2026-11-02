"""Tests for nlpath.mapgen: structural properties of generated maps."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nlpath.grid import GridMap
from nlpath.mapgen import (SMALL, TIERS, benchmark_maps, generate_map,
                           is_corner_separated, small_maps)
from reference import make_map

MAPS_DIR = Path(__file__).resolve().parents[1] / "data" / "maps"
ALL_PARAMS = [*TIERS.values(), SMALL]

seeds = st.integers(min_value=0, max_value=10**6)
params = st.sampled_from(ALL_PARAMS)

ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))


def component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """4-connected component of ``mask`` containing ``start`` (BFS)."""
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ORTHO:
            n = (r + dr, c + dc)
            if (0 <= n[0] < mask.shape[0] and 0 <= n[1] < mask.shape[1]
                    and mask[n] and n not in seen):
                seen.add(n)
                queue.append(n)
    return seen


def entrances(gm: GridMap, name: str) -> set[tuple[int, int]]:
    """Free cells outside region ``name`` that are 4-adjacent to it."""
    mask = gm.region_mask(name)
    out = set()
    for r, c in gm.region_cells(name):
        for dr, dc in ORTHO:
            n = (r + dr, c + dc)
            if gm.free[n] and not mask[n]:
                out.add(n)
    return out


@given(seed=seeds, p=params)
def test_same_seed_same_map(seed, p):
    assert generate_map("m", seed, p) == generate_map("m", seed, p)


@given(seed=seeds, p=params)
def test_structure(seed, p):
    gm = generate_map("m", seed, p)
    assert gm.shape == (p.height, p.width)
    # Border is wall.
    assert not gm.free[[0, -1], :].any() and not gm.free[:, [0, -1]].any()
    # All free cells are mutually reachable. With no corner cutting (plan §2.5),
    # 4-connectivity is exactly reachability by legal moves.
    first = tuple(int(x) for x in np.argwhere(gm.free)[0])
    assert len(component(gm.free, first)) == gm.free.sum()
    # Regions: corridor plus n_named rooms, each non-empty and contiguous.
    assert gm.region_names[0] == "corridor"
    assert len(gm.region_names) == 1 + min(p.n_named, gm.meta["n_rooms"])
    for name in gm.region_names:
        cells = gm.region_cells(name)
        assert cells
        assert len(component(gm.region_mask(name), cells[0])) == len(cells)


@given(seed=seeds, p=params)
def test_has_named_pass_through_room(seed, p):
    """Some named room has two entrances, so a prohibition can force a reroute."""
    gm = generate_map("m", seed, p)
    rooms = [n for n in gm.region_names if n != "corridor"]
    assert any(len(entrances(gm, n)) >= 2 for n in rooms)


@given(seed=seeds, p=params)
def test_inner_regions_have_no_door_on_main_corridor(seed, p):
    gm = generate_map("m", seed, p)
    spine = gm.region_mask("corridor")
    for name in gm.meta["inner_regions"]:
        for r, c in entrances(gm, name):
            assert not any(spine[r + dr, c + dc] for dr, dc in ORTHO)


@given(seed=seeds, p=params)
def test_generated_maps_are_corner_separated(seed, p):
    """Precondition of T6 (docs/soundness.md §6)."""
    assert is_corner_separated(generate_map("m", seed, p))


def test_benchmark_maps_are_corner_separated():
    assert all(is_corner_separated(gm) for gm in benchmark_maps() + small_maps())


def test_corner_separation_detects_a_grazed_corner():
    # (1,1) -> (2,2) passes the corner of lab cell (1,2) without entering it.
    assert not is_corner_separated(make_map(["####", "#.L#", "#..#", "####"]))
    # Entering the region on the diagonal itself is fine.
    assert is_corner_separated(make_map(["####", "#.L#", "#.L#", "####"]))
    # A wall in the 2x2 block makes the diagonal illegal anyway.
    assert is_corner_separated(make_map(["####", "#.L#", "##.#", "####"]))


def test_medium_and_hard_maps_have_inner_regions():
    for gm in benchmark_maps():
        if gm.meta["tier"] in ("medium", "hard"):
            assert gm.meta["inner_regions"], gm.map_id


def test_benchmark_sets():
    big, small = benchmark_maps(), small_maps()
    assert len(big) == 20 and len(small) == 10
    assert all(gm.shape == (40, 40) for gm in big)
    assert all(gm.shape == (15, 15) for gm in small)
    ids = [gm.map_id for gm in big + small]
    assert len(set(ids)) == len(ids)


@pytest.mark.skipif(not MAPS_DIR.exists(), reason="maps not generated yet")
def test_committed_maps_match_generator():
    """data/maps/*.json is exactly what scripts/make_maps.py regenerates."""
    for gm in benchmark_maps() + small_maps():
        assert GridMap.load(MAPS_DIR / f"{gm.map_id}.json") == gm, gm.map_id
