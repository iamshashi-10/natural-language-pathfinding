"""Tests for nlpath.astar on hand-checked cases, plus a networkx cross-check.

Expected costs in the hand-checked tests are worked out in the comments.
"""

import math

import networkx as nx
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nlpath.astar import (SQRT2, astar, grid_moves, octile, octile_field,
                          plan_grid)
from nlpath.mapgen import benchmark_maps

R2 = SQRT2


def grid(rows: list[str]) -> np.ndarray:
    """'#' = not allowed, anything else = allowed."""
    return np.array([[ch != "#" for ch in row] for row in rows])


def goal_at(shape, *cells) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def ones(shape) -> np.ndarray:
    return np.ones(shape)


def is_valid_8_path(allowed, path) -> bool:
    """Every step is a legal move (independent of grid_moves)."""
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        if max(abs(dr), abs(dc)) != 1 or not allowed[r1, c1]:
            return False
        if dr and dc and not (allowed[r0, c1] and allowed[r1, c0]):
            return False
    return all(allowed[c] for c in path)


def cost_of(path, weights) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) * weights[b]
               for a, b in zip(path, path[1:]))


# ----- geometry ----------------------------------------------------------------


def test_octile():
    assert octile((0, 0), (0, 0)) == 0
    assert octile((0, 0), (0, 5)) == 5
    assert octile((0, 0), (3, 3)) == pytest.approx(3 * R2)
    assert octile((0, 0), (2, 5)) == pytest.approx(3 + 2 * R2)


def test_octile_field_takes_nearest_goal():
    field = octile_field((1, 5), [(0, 0), (0, 4)])
    assert field.tolist() == [[0, 1, 2, 1, 0]]
    assert np.isinf(octile_field((2, 2), [])).all()


def test_moves_in_open_space():
    moves = dict(grid_moves(grid(["...", "...", "..."]), (1, 1)))
    assert len(moves) == 8
    assert moves[(0, 1)] == 1.0 and moves[(0, 0)] == R2


def test_no_corner_cutting_one_side_blocked():
    # From (1,0) to (0,1): orthogonal neighbours are (0,0) '#' and (1,1) '.'.
    # One blocked orthogonal is enough to forbid the diagonal.
    allowed = grid(["#.", ".."])
    assert (0, 1) not in dict(grid_moves(allowed, (1, 0)))


def test_no_corner_cutting_path():
    # . #
    # . .     (0,0) -> (1,1): the diagonal is cut by the wall at (0,1),
    #         so the path is (0,0),(1,0),(1,1) with cost 2 instead of sqrt 2.
    allowed = grid([".#", ".."])
    res = plan_grid(allowed, ones((2, 2)), (0, 0), goal_at((2, 2), (1, 1)))
    assert res.path == [(0, 0), (1, 0), (1, 1)]
    assert res.cost == pytest.approx(2.0)


def test_diagonal_between_two_walls_is_impossible():
    # . #
    # # .     No orthogonal route and the diagonal squeezes between walls.
    allowed = grid([".#", "#."])
    res = plan_grid(allowed, ones((2, 2)), (0, 0), goal_at((2, 2), (1, 1)))
    assert not res.found and res.reason == "goal unreachable"


# ----- hand-checked optimal paths ---------------------------------------------


def test_open_grid_diagonal():
    # 3x3 open grid, corner to corner: two diagonal steps, cost 2*sqrt2.
    res = plan_grid(grid(["..."] * 3), ones((3, 3)), (0, 0), goal_at((3, 3), (2, 2)))
    assert res.path == [(0, 0), (1, 1), (2, 2)]
    assert res.cost == pytest.approx(2 * R2)


def test_open_grid_knight_move_cost():
    # (0,0) -> (1,4) on an open grid: 1 diagonal + 3 straight = sqrt2 + 3.
    res = plan_grid(grid(["....."] * 2), ones((2, 5)), (0, 0), goal_at((2, 5), (1, 4)))
    assert res.cost == pytest.approx(R2 + 3)
    assert len(res.path) == 5


def test_wall_forces_detour():
    # . . . . .
    # . # # # .     start (2,0), goal (2,4); the pocket (2,2) is a dead end.
    # . # . # .
    # The diagonals (1,0)->(0,1) and (0,3)->(1,4) would cut the corner at
    # (1,1) / (1,3), so the path goes all the way round: 8 orthogonal steps.
    allowed = grid([".....", ".###.", ".#.#."])
    res = plan_grid(allowed, ones((3, 5)), (2, 0), goal_at((3, 5), (2, 4)))
    assert res.cost == pytest.approx(8.0)
    assert res.path == [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                        (1, 4), (2, 4)]


def test_preferred_row_is_taken_when_detour_is_small():
    # Row 0 has weight 0.5. Start (1,0), goal (1,4).
    # Straight along row 1: 4 * 1.0 = 4.0.
    # Via row 0: diag up to (0,1) 0.5*R2, steps to (0,2),(0,3) 2*0.5, diag down
    # into (1,4) R2*1.0 = 0.707 + 1.0 + 1.414 = 3.121 -> cheaper, so it is taken.
    weights = np.array([[0.5] * 5, [1.0] * 5])
    res = plan_grid(grid(["....."] * 2), weights, (1, 0), goal_at((2, 5), (1, 4)))
    assert res.cost == pytest.approx(0.5 * R2 + 1.0 + R2)
    assert res.path == [(1, 0), (0, 1), (0, 2), (0, 3), (1, 4)]


def test_preferred_row_is_abandoned_when_detour_is_large():
    # Same idea, but the preferred row 0 is 3 rows away from start/goal (row 3).
    # Straight along row 3: 4.0.
    # Touching row 0 means entering rows 2, 1, 0 on the way up and 1, 2, 3 on
    # the way down: cost >= 1 + 1 + 0.5 + 1 + 1 + 1 = 5.5 > 4, so it is not used.
    weights = np.ones((4, 7))
    weights[0] = 0.5
    res = plan_grid(grid(["......."] * 4), weights, (3, 0), goal_at((4, 7), (3, 4)))
    assert res.cost == pytest.approx(4.0)
    assert all(r == 3 for r, _ in res.path)


def test_avoided_cell_is_paid_for_or_bypassed():
    # . . 3 . .    Column 2 has weight 3 and spans the whole grid, so it must be
    # . . 3 . .    crossed. Cheapest is straight along row 1, entering
    # . . 3 . .    (1,1),(1,2),(1,3),(1,4): 1 + 3 + 1 + 1 = 6.
    weights = np.ones((3, 5))
    weights[:, 2] = 3.0
    res = plan_grid(grid(["....."] * 3), weights, (1, 0), goal_at((3, 5), (1, 4)))
    assert res.cost == pytest.approx(6.0)
    # Add an unpenalised row 3 below: (1,0)->(2,1)->(3,2)->(2,3)->(1,4) is four
    # diagonals = 4*R2 = 5.657 < 6, so the bypass is taken.
    weights = np.vstack([weights, np.ones((1, 5))])
    res = plan_grid(grid(["....."] * 4), weights, (1, 0), goal_at((4, 5), (1, 4)))
    assert res.cost == pytest.approx(4 * R2)
    assert (3, 2) in res.path


def test_nearest_of_several_goals():
    res = plan_grid(grid(["......."]), ones((1, 7)), (0, 2), goal_at((1, 7), (0, 0), (0, 6)))
    assert res.path == [(0, 2), (0, 1), (0, 0)]
    assert res.cost == pytest.approx(2.0)


def test_start_already_in_goal():
    res = plan_grid(grid(["..."]), ones((1, 3)), (0, 1), goal_at((1, 3), (0, 1)))
    assert res.path == [(0, 1)] and res.cost == 0


def test_no_path_reasons():
    allowed = grid(["..#.."])
    w = ones((1, 5))
    assert plan_grid(allowed, w, (0, 0), goal_at((1, 5), (0, 4))).reason == "goal unreachable"
    assert plan_grid(allowed, w, (0, 2), goal_at((1, 5), (0, 4))).reason == \
        "start cell is not allowed"
    assert plan_grid(allowed, w, (0, 0), goal_at((1, 5), (0, 2))).reason == \
        "no allowed goal cell"


def test_rejects_non_positive_weights():
    with pytest.raises(ValueError, match="positive"):
        plan_grid(grid(["..."]), np.array([[1.0, 0.0, 1.0]]), (0, 0), goal_at((1, 3), (0, 2)))


# ----- heuristic scaling (the v1 bug, plan §2.5) -------------------------------

# Found by exhaustive search over 3x4 grids with weights in {0.5, 1}.
SCALING_WEIGHTS = np.array([[0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 1.0, 0.5],
                            [0.5, 0.5, 1.0, 0.5]])


def test_scaled_heuristic_is_optimal_where_unscaled_is_not():
    allowed = np.ones((3, 4), dtype=bool)
    goals = goal_at((3, 4), (2, 3))
    # Optimal: (0,1),(0,2) at 0.5 each, diag to (1,3) at 0.5*R2, down to (2,3) 0.5.
    optimal = 0.5 + 0.5 + 0.5 * R2 + 0.5
    res = plan_grid(allowed, SCALING_WEIGHTS, (0, 0), goals)
    assert res.cost == pytest.approx(optimal)

    # Same search with the unscaled octile heuristic (inadmissible): worse path.
    h = octile_field((3, 4), [(2, 3)])
    bad = astar((0, 0), lambda v: bool(goals[v]),
                lambda v: ((u, s * SCALING_WEIGHTS[u]) for u, s in grid_moves(allowed, v)),
                lambda v: float(h[v]))
    assert bad.cost > optimal + 1e-9


# ----- node counting -----------------------------------------------------------


def test_heuristic_reduces_expansions():
    allowed = np.ones((20, 20), dtype=bool)
    goals = goal_at((20, 20), (19, 19))
    w = ones((20, 20))
    with_h = plan_grid(allowed, w, (0, 0), goals)
    dijkstra = astar((0, 0), lambda v: bool(goals[v]),
                     lambda v: grid_moves(allowed, v), lambda v: 0.0)
    assert with_h.cost == pytest.approx(dijkstra.cost) == pytest.approx(19 * R2)
    assert 0 < with_h.nodes_expanded < dijkstra.nodes_expanded


def test_unreachable_counts_whole_component():
    # Start component has 3 cells; all are expanded before giving up.
    res = plan_grid(grid(["...#."]), ones((1, 5)), (0, 0), goal_at((1, 5), (0, 4)))
    assert not res.found and res.nodes_expanded == 3


# ----- cross-check against networkx (reference implementation) ----------------


def reference_cost(allowed, weights, start, goals) -> float:
    """Dijkstra in networkx on an explicitly built graph — written without grid_moves."""
    g = nx.DiGraph()
    h, w = allowed.shape
    for r in range(h):
        for c in range(w):
            if not allowed[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr, dc) == (0, 0) or not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not allowed[nr, nc]:
                        continue
                    if dr and dc and not (allowed[r, nc] and allowed[nr, c]):
                        continue
                    g.add_edge((r, c), (nr, nc), weight=math.hypot(dr, dc) * weights[nr, nc])
    g.add_node(start)
    targets = [tuple(map(int, x)) for x in np.argwhere(goals & allowed)]
    dist = nx.single_source_dijkstra_path_length(g, start)
    return min((dist[t] for t in targets if t in dist), default=math.inf)


@st.composite
def grid_problems(draw):
    h = draw(st.integers(2, 10))
    w = draw(st.integers(2, 10))
    cells = st.lists(st.booleans(), min_size=h * w, max_size=h * w)
    allowed = np.array(draw(cells)).reshape(h, w)
    weights = np.array(draw(st.lists(st.sampled_from([0.5, 1.0, 3.0]),
                                     min_size=h * w, max_size=h * w))).reshape(h, w)
    start = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    allowed[start] = True
    goals = np.array(draw(cells)).reshape(h, w)
    return allowed, weights, start, goals


@given(grid_problems())
def test_matches_networkx_on_random_grids(problem):
    allowed, weights, start, goals = problem
    res = plan_grid(allowed, weights, start, goals)
    ref = reference_cost(allowed, weights, start, goals)
    if math.isinf(ref):
        assert not res.found
    else:
        assert res.found and res.cost == pytest.approx(ref)
        assert res.path[0] == start and goals[res.path[-1]]
        assert is_valid_8_path(allowed, res.path)
        assert cost_of(res.path, weights) == pytest.approx(res.cost)


def test_matches_networkx_on_benchmark_maps():
    rng = np.random.default_rng(0)
    for gm in benchmark_maps():
        free_cells = np.argwhere(gm.free)
        weights = np.where(gm.region_mask("corridor"), 0.5, 1.0)
        for _ in range(3):
            start = tuple(map(int, free_cells[rng.integers(len(free_cells))]))
            goal_region = gm.region_names[1 + rng.integers(len(gm.region_names) - 1)]
            goals = gm.region_mask(goal_region)
            res = plan_grid(gm.free, weights, start, goals)
            assert res.cost == pytest.approx(reference_cost(gm.free, weights, start, goals))
            assert is_valid_8_path(gm.free, res.path)
