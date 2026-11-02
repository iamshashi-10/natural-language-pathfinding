"""Room / corridor / doorway map generator.

Algorithm (binary space partitioning):

1. Start from a grid of walls.
2. Split the interior recursively. The first ``corridor_depth`` levels of splits
   become corridors (wall | corridor | wall); deeper splits become a single wall
   between two rooms. Each corridor is extended one cell at both ends so it joins
   its parent corridor, which makes the corridor network connected.
3. The leaves of the partition are rooms.
4. Doorways (1 cell, unlabelled) are carved through walls:
   - every room next to a corridor gets a corridor door, except "inner" rooms,
     which are reachable only through another room;
   - any room still disconnected gets a door to a connected neighbour;
   - extra room-to-room and second corridor doors create pass-through rooms;
   - at least one room is guaranteed doors to two different spaces.
5. Up to ``n_named`` rooms receive names from the vocabulary; the top-level
   corridor is labelled ``corridor``. Side corridors, doorways and the remaining
   rooms stay unlabelled.

Everything is driven by one ``numpy.random.Generator``, so a seed fully
determines the map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from nlpath.grid import NO_LABEL, REGION_VOCAB, GridMap

ROOM_NAMES: tuple[str, ...] = tuple(n for n in REGION_VOCAB if n != "corridor")

CORRIDOR_SPACE = 0  # space id of the corridor network; rooms are 1..n


@dataclass(frozen=True)
class GenParams:
    """Parameters of one generated map (stored in the map's ``meta``)."""

    height: int = 40
    width: int = 40
    corridor_depth: int = 1     # levels of BSP splits that become corridors
    corridor_width: int = 2
    min_block: int = 8          # minimum side of a block next to a corridor
    min_room: int = 4           # minimum room side (free cells)
    max_room: int = 12          # rooms are split until both sides are <= this
    n_named: int = 5            # number of rooms that receive a region name
    n_inner: int = 0            # rooms with no corridor door
    p_extra_door: float = 0.2   # chance of a door between two adjacent rooms
    p_second_door: float = 0.2  # chance of a second corridor door for a room


# Complexity tiers for the 40×40 benchmark maps (plan §2.6: "varying complexity").
TIERS: dict[str, GenParams] = {
    "easy": GenParams(corridor_depth=1, max_room=14, n_named=4, n_inner=0,
                      p_extra_door=0.1, p_second_door=0.2),
    "medium": GenParams(corridor_depth=2, max_room=11, n_named=6, n_inner=1,
                        p_extra_door=0.2, p_second_door=0.2),
    "hard": GenParams(corridor_depth=2, max_room=9, n_named=7, n_inner=2,
                      p_extra_door=0.3, p_second_door=0.3),
}
SMALL = GenParams(height=15, width=15, corridor_depth=1, corridor_width=1,
                  min_block=3, min_room=3, max_room=6, n_named=3, n_inner=0,
                  p_extra_door=0.2, p_second_door=0.2)

Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive


# ----- partition -------------------------------------------------------------


def _split_point(rng: np.random.Generator, lo: int, hi: int, gap: int,
                 min_side: int) -> int | None:
    """Pick s so that [lo, s-1] and [s+gap, hi] both have >= min_side cells."""
    first, last = lo + min_side, hi - min_side - gap + 1
    if first > last:
        return None
    return int(rng.integers(first, last + 1))


def _partition(rng: np.random.Generator, p: GenParams, free: np.ndarray,
               corridor: np.ndarray, spine: np.ndarray) -> list[Rect]:
    """Carve corridors into ``free``/``corridor``/``spine``; return room rects."""
    rooms: list[Rect] = []

    def split(rect: Rect, depth: int) -> None:
        r0, c0, r1, c1 = rect
        h, w = r1 - r0 + 1, c1 - c0 + 1
        is_corridor = depth < p.corridor_depth
        if not is_corridor and max(h, w) <= p.max_room:
            rooms.append(rect)
            return
        gap = p.corridor_width + 2 if is_corridor else 1
        min_side = p.min_block if is_corridor else p.min_room
        horizontal = h > w or (h == w and bool(rng.integers(2)))
        lo, hi = (r0, r1) if horizontal else (c0, c1)
        s = _split_point(rng, lo, hi, gap, min_side)
        if s is None and is_corridor:  # block too small for a corridor
            split(rect, p.corridor_depth)
            return
        if s is None:  # too small to split further
            rooms.append(rect)
            return
        if is_corridor:
            a, b = s + 1, s + p.corridor_width  # corridor rows/cols, inclusive
            # Extend one cell past each end to join the parent corridor
            # (the border itself is restored to wall afterwards).
            if horizontal:
                cells = (slice(a, b + 1), slice(c0 - 1, c1 + 2))
            else:
                cells = (slice(r0 - 1, r1 + 2), slice(a, b + 1))
            free[cells] = True
            corridor[cells] = True
            if depth == 0:
                spine[cells] = True
        if horizontal:
            split((r0, c0, s - 1, c1), depth + 1)
            split((s + gap, c0, r1, c1), depth + 1)
        else:
            split((r0, c0, r1, s - 1), depth + 1)
            split((r0, s + gap, r1, c1), depth + 1)

    split((1, 1, p.height - 2, p.width - 2), 0)
    return rooms


# ----- doors -----------------------------------------------------------------


def _door_candidates(space: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map each pair of spaces (a < b) to the wall cells that separate them.

    A wall cell is a door candidate between spaces a and b if its two opposite
    orthogonal neighbours (N/S or W/E) belong to a and b.
    """
    h, w = space.shape
    cands: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if space[r, c] != -1:
                continue
            for (ra, ca), (rb, cb) in (((r - 1, c), (r + 1, c)),
                                       ((r, c - 1), (r, c + 1))):
                a, b = int(space[ra, ca]), int(space[rb, cb])
                if a >= 0 and b >= 0 and a != b:
                    cands.setdefault((min(a, b), max(a, b)), []).append((r, c))
    return cands


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _place_doors(rng: np.random.Generator, p: GenParams, space: np.ndarray,
                 n_rooms: int) -> tuple[list[tuple[int, int]], set[int]]:
    """Choose doorway cells. Returns (doors, inner room ids)."""
    cands = _door_candidates(space)
    pairs = sorted(cands)
    neighbours = {i: set() for i in range(n_rooms + 1)}
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)

    uf = _UnionFind(n_rooms + 1)
    doors: list[tuple[int, int]] = []
    door_pairs: set[tuple[int, int]] = set()

    def add_door(a: int, b: int) -> None:
        pair = (min(a, b), max(a, b))
        options = [cell for cell in cands[pair] if cell not in doors]
        # Avoid doors adjacent to an existing door on the same wall.
        spaced = [cell for cell in options
                  if all(abs(cell[0] - d[0]) + abs(cell[1] - d[1]) > 1 for d in doors)]
        options = spaced or options
        if not options:
            return
        doors.append(options[int(rng.integers(len(options)))])
        door_pairs.add(pair)
        uf.union(a, b)

    rooms = list(range(1, n_rooms + 1))
    # Inner rooms: no corridor door, but they must have a room neighbour.
    inner_ok = [r for r in rooms if neighbours[r] - {CORRIDOR_SPACE}]
    k = min(p.n_inner, len(inner_ok))
    inner = {int(x) for x in rng.choice(inner_ok, size=k, replace=False)} if k else set()

    for r in rooms:
        if r not in inner and CORRIDOR_SPACE in neighbours[r]:
            add_door(r, CORRIDOR_SPACE)

    # Connect anything still disconnected through a connected neighbour.
    while True:
        root = uf.find(CORRIDOR_SPACE)
        loose = [r for r in rooms if uf.find(r) != root]
        if not loose:
            break
        progressed = False
        for r in loose:
            linked = sorted(nb for nb in neighbours[r] if uf.find(nb) == root)
            room_links = [nb for nb in linked if nb != CORRIDOR_SPACE]
            linked = room_links or linked  # inner rooms prefer a room
            if linked:
                add_door(r, linked[int(rng.integers(len(linked)))])
                progressed = True
        if not progressed:  # isolated pocket; give up on it being inner
            r = loose[0]
            if not neighbours[r]:
                raise RuntimeError(f"room {r} has no neighbour to connect to")
            inner.discard(r)
            add_door(r, sorted(neighbours[r])[0])

    # Extra doors make cut-through routes (prohibitions then cause reroutes).
    for a, b in pairs:
        if (a, b) in door_pairs:
            continue
        if a == CORRIDOR_SPACE:
            continue  # corridor doors handled below
        if rng.random() < p.p_extra_door:
            add_door(a, b)
    for r in rooms:
        if r not in inner and CORRIDOR_SPACE in neighbours[r] \
                and rng.random() < p.p_second_door:
            add_door(r, CORRIDOR_SPACE)

    # Guarantee at least one pass-through room (doors to two different spaces).
    if not any(len(_door_spaces(r, doors, space)) >= 2 for r in rooms):
        for a, b in pairs:
            if a != CORRIDOR_SPACE and (a, b) not in door_pairs:
                add_door(a, b)
                break
    return doors, inner


def _door_spaces(room: int, doors: list[tuple[int, int]], space: np.ndarray) -> set[int]:
    """Spaces other than ``room`` that ``room`` has a door to."""
    out = set()
    for r, c in doors:
        for (ra, ca), (rb, cb) in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
            a, b = int(space[ra, ca]), int(space[rb, cb])
            if room in (a, b) and a >= 0 and b >= 0:
                out.add(b if a == room else a)
    return out


# ----- naming ----------------------------------------------------------------


def _choose_named(rng: np.random.Generator, p: GenParams, n_rooms: int,
                  inner: set[int], doors: list[tuple[int, int]],
                  space: np.ndarray) -> list[int]:
    """Pick the rooms that get names.

    Priority: inner rooms and their gateways, then (if none of those already is
    one) a pass-through room, so that a prohibition can force a reroute.
    """
    n = min(p.n_named, n_rooms, len(ROOM_NAMES))
    chosen: list[int] = []
    for r in sorted(inner):
        gateways = sorted(_door_spaces(r, doors, space) - {CORRIDOR_SPACE})
        for x in [r, *gateways[:1]]:
            if x not in chosen and len(chosen) < n:
                chosen.append(x)
    passthrough = [r for r in range(1, n_rooms + 1)
                   if len(_door_spaces(r, doors, space)) >= 2]
    if passthrough and not set(chosen) & set(passthrough) and len(chosen) < n:
        chosen.append(passthrough[int(rng.integers(len(passthrough)))])
    rest = [r for r in range(1, n_rooms + 1) if r not in chosen]
    extra = rng.choice(rest, size=n - len(chosen), replace=False) if n > len(chosen) else []
    return chosen + [int(x) for x in extra]


# ----- public API ------------------------------------------------------------


def generate_map(map_id: str, seed: int, params: GenParams, tier: str = "") -> GridMap:
    """Generate one map. The same (seed, params) always gives the same map."""
    rng = np.random.default_rng(seed)
    p = params
    shape = (p.height, p.width)
    free = np.zeros(shape, dtype=bool)
    corridor = np.zeros(shape, dtype=bool)
    spine = np.zeros(shape, dtype=bool)

    room_rects = _partition(rng, p, free, corridor, spine)
    for arr in (free, corridor, spine):  # border is always wall
        arr[[0, -1], :] = False
        arr[:, [0, -1]] = False

    space = np.full(shape, -1, dtype=np.int16)  # -1 = wall
    space[corridor] = CORRIDOR_SPACE
    for i, (r0, c0, r1, c1) in enumerate(room_rects, start=1):
        free[r0:r1 + 1, c0:c1 + 1] = True
        space[r0:r1 + 1, c0:c1 + 1] = i

    doors, inner = _place_doors(rng, p, space, len(room_rects))
    for cell in doors:
        free[cell] = True

    named = _choose_named(rng, p, len(room_rects), inner, doors, space)
    names = [str(x) for x in rng.permutation(ROOM_NAMES)[:len(named)]]
    region_names = tuple(["corridor", *names])
    labels = np.full(shape, NO_LABEL, dtype=np.int8)
    labels[spine] = 0
    for idx, room in enumerate(named, start=1):
        labels[space == room] = idx

    no_corridor_door = {r for r in range(1, len(room_rects) + 1)
                        if CORRIDOR_SPACE not in _door_spaces(r, doors, space)}
    inner_names = sorted(region_names[i] for i, r in enumerate(named, start=1)
                         if r in no_corridor_door)
    meta = {
        "seed": seed,
        "tier": tier,
        "params": asdict(p),
        "n_rooms": len(room_rects),
        # Named rooms reachable only through another room (useful for Day 7's
        # infeasible instructions; feasibility itself comes from networkx).
        "inner_regions": inner_names,
    }
    return GridMap(map_id, free, labels, region_names, meta)


def is_corner_separated(gm: GridMap) -> bool:
    """True if no physically legal diagonal move grazes a region it does not enter.

    That is: for every diagonal v -> u with both corner cells free, a corner cell
    in region r implies v or u is in r. Theorem T6 (docs/soundness.md §6) needs
    this; rooms enclosed by walls with 1-cell doors satisfy it.
    """
    h, w = gm.shape
    free, labels = gm.free, gm.labels
    for r in range(h - 1):
        for c in range(w - 1):
            block = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if not all(free[x] for x in block):
                continue  # no legal diagonal inside this 2x2 block
            for v, u, corners in (((r, c), (r + 1, c + 1), ((r, c + 1), (r + 1, c))),
                                  ((r, c + 1), (r + 1, c), ((r, c), (r + 1, c + 1)))):
                for corner in corners:
                    lab = labels[corner]
                    if lab != NO_LABEL and labels[v] != lab and labels[u] != lab:
                        return False
    return True


BASE_SEED = 1000
SMALL_BASE_SEED = 2000
N_MAPS = 20
N_SMALL = 10


def benchmark_tier(i: int) -> str:
    """Tier of benchmark map ``i``: 6 easy, 7 medium, 7 hard."""
    return "easy" if i < 6 else "medium" if i < 13 else "hard"


def benchmark_maps() -> list[GridMap]:
    """The 20 40×40 benchmark maps (plan §2.6)."""
    return [generate_map(f"map_{i:02d}", BASE_SEED + i, TIERS[benchmark_tier(i)],
                         benchmark_tier(i))
            for i in range(N_MAPS)]


def small_maps() -> list[GridMap]:
    """The 15×15 maps (plan §6: M1 is also reported on small grids)."""
    return [generate_map(f"small_{i:02d}", SMALL_BASE_SEED + i, SMALL, "small")
            for i in range(N_SMALL)]
