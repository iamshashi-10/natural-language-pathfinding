"""The instruction benchmark (plan §4.1): data format, gold LTL, generation.

Each instruction has three wordings of the same request, a map and a start
cell, the gold intent form, the gold LTLf formula (from T6, docs/soundness.md
§6), machine-checkable checks, and a ``feasible`` label computed by the
networkx reference solver — never by our planner.

Generation is deterministic (fixed seed). Start cells are free, unlabelled
cells, so they lie outside every region an instruction mentions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nlpath.grid import NO_LABEL, Cell, GridMap
from nlpath.intent import IntentForm, validate
from nlpath.phrasings import HANDWRITTEN, TEMPLATES
from nlpath.reference import reference_path, reference_solve
from nlpath.verify import hard_checks, path_length

CATEGORIES = ("plain", "prohibition", "stop", "ordering", "soft", "infeasible")

# (category, shape, count, template wordings) for the templated part. Together
# with 5 hand-written patterns per category this gives plan §4.1's counts:
# 20 / 35 / 30 / 30 / 35 / 15 = 165 instructions.
TEMPLATE_PLAN = [
    ("plain", "G", 15, "plain"),
    ("prohibition", "G X", 30, "prohibition"),
    ("stop", "G W", 25, "stop"),
    ("ordering", "G A B", 25, "ordering"),
    ("soft", "G C", 20, "soft_prefer"),
    ("soft", "G Y", 10, "soft_avoid"),
    ("infeasible", "G X", 7, "prohibition"),
    ("infeasible", "G W", 3, "stop"),   # infeasible because of strict ordering
]
SMALL_PLAN = [
    ("plain", "G", 5, "plain"),
    ("prohibition", "G X", 5, "prohibition"),
    ("stop", "G W", 5, "stop"),
    ("ordering", "G A B", 5, "ordering"),
]
BENCHMARK_SEED = 3000
MAX_TRIES = 300


@dataclass(frozen=True)
class Instruction:
    """One benchmark instruction (three wordings of the same request)."""

    id: str
    category: str
    source: str                 # "template" or "handwritten"
    map_id: str
    start: Cell
    texts: tuple[str, ...]      # original + two paraphrases
    gold_intent: IntentForm
    gold_ltl: str
    feasible: bool
    ref_cost: float | None      # optimal cost J from the reference; None if infeasible
    binding: bool               # constraints change the route: hard ones raise the
                                # optimal cost; soft ones make the optimum a detour
    checks: tuple[dict, ...] = field(default=())

    def to_dict(self) -> dict:
        return {"id": self.id, "category": self.category, "source": self.source,
                "map_id": self.map_id, "start": list(self.start),
                "texts": list(self.texts), "feasible": self.feasible,
                "gold_intent": self.gold_intent.to_dict(), "gold_ltl": self.gold_ltl,
                "ref_cost": self.ref_cost, "binding": self.binding,
                "checks": list(self.checks)}

    @classmethod
    def from_dict(cls, d: dict) -> Instruction:
        return cls(d["id"], d["category"], d["source"], d["map_id"], tuple(d["start"]),
                   tuple(d["texts"]), IntentForm.from_dict(d["gold_intent"]),
                   d["gold_ltl"], d["feasible"], d["ref_cost"], d["binding"],
                   tuple(d["checks"]))


def save_benchmark(items: list[Instruction], path: str | Path) -> None:
    Path(path).write_text(json.dumps([it.to_dict() for it in items], indent=1) + "\n")


def load_benchmark(path: str | Path) -> list[Instruction]:
    return [Instruction.from_dict(d) for d in json.loads(Path(path).read_text())]


# ----- gold representations ----------------------------------------------------


def _disj(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else "(" + " | ".join(names) + ")"


def gold_ltl(intent: IntentForm) -> str:
    """φ_I of theorem T6 (soft preferences have no LTL counterpart and are dropped).

    psi_{m+1} = F(g & last);  psi_j = !(u_{j+1} | … ) U (u_j & psi_{j+1});
    phi = G !(x_1 | …) & psi_1.  ``last`` marks the final position of the trace.
    """
    seq = intent.sequence
    psi = f"F({seq[-1]} & last)"
    for j in range(len(seq) - 2, -1, -1):
        psi = f"!{_disj(seq[j + 1:])} U ({seq[j]} & {psi})"
        psi = f"({psi})"
    if intent.forbidden_regions:
        return f"G !{_disj(intent.forbidden_regions)} & {psi}"
    return psi


def instruction_checks(intent: IntentForm) -> list[dict]:
    """Hard checks (pass/fail) plus a fraction_in_region per soft region."""
    soft = [*intent.preferred_regions, *intent.avoided_regions]
    return hard_checks(intent) + [{"type": "fraction_in_region", "region": r} for r in soft]


# ----- generation --------------------------------------------------------------


def display(name: str) -> str:
    """How a region is written in text: ``store_room`` -> ``store room``."""
    return name.replace("_", " ")


def intent_from_slots(shape: str, slots: dict[str, str]) -> IntentForm:
    letters = shape.split()
    return IntentForm(
        goal_region=slots["G"],
        forbidden_regions=tuple(slots[x] for x in ("X", "X2") if x in letters),
        ordered_waypoints=tuple(slots[x] for x in ("W", "A", "B") if x in letters),
        preferred_regions=("corridor",) if "C" in letters else (),
        avoided_regions=(slots["Y"],) if "Y" in letters else ())


def _choose_slots(rng, gm: GridMap, shape: str, want_feasible: bool) -> dict[str, str] | None:
    rooms = [n for n in gm.legend() if n != "corridor"]
    letters = [s for s in shape.split() if s != "C"]
    if len(letters) > len(rooms):
        return None
    picked = [str(x) for x in rng.permutation(rooms)[:len(letters)]]
    slots = dict(zip(letters, picked))
    inner = gm.meta.get("inner_regions", [])
    if not want_feasible and inner:
        # Infeasible: put the deepest element (goal, or the waypoint) in a room
        # reachable only through another room, which is then likely forbidden.
        key = "W" if "W" in slots and "X" not in slots else "G"
        target = str(rng.choice(inner))
        other = next((k for k, v in slots.items() if v == target), None)
        if other is not None:
            slots[other] = slots[key]
        slots[key] = target
    if "C" in shape.split():
        slots["C"] = "corridor"
    return slots


def _random_start(rng, gm: GridMap) -> Cell:
    cells = np.argwhere(gm.free & (gm.labels == NO_LABEL))
    r, c = cells[rng.integers(len(cells))]
    return int(r), int(c)


def _is_binding(gm: GridMap, intent: IntentForm, start: Cell, cost: float) -> bool:
    """Do the hard constraints raise the optimal cost above "just go to the goal"?"""
    free_cost = reference_solve(gm, IntentForm(intent.goal_region,
                                               preferred_regions=intent.preferred_regions,
                                               avoided_regions=intent.avoided_regions),
                                start)
    return bool(cost > free_cost + 1e-9)


def _soft_detour(gm: GridMap, intent: IntentForm, start: Cell) -> bool:
    """Does honouring the soft preference make the optimal path longer than the
    shortest path satisfying the hard constraints? (The trade-off is then visible.)"""
    _, path = reference_path(gm, intent, start)
    return bool(path is not None) and bool(
        path_length(path) > reference_solve(gm, intent.hard_part(), start) + 1e-9)


def make_instruction(rng, maps: list[GridMap], first_map: int, category: str,
                     shape: str, wordings: list[str], source: str,
                     ident: str, alt_rng=None) -> Instruction:
    """Sample map/start/regions until the reference solver gives the wanted label.

    Main categories need a feasible instance (prohibitions: preferably one
    where the prohibition changes the optimal route); ``infeasible`` needs an
    infeasible one.
    """
    want_feasible = category != "infeasible"
    want_binding = "X" in shape.split() and want_feasible
    fallback = None
    for attempt in range(MAX_TRIES):
        gm = maps[(first_map + attempt) % len(maps)]
        slots = _choose_slots(rng, gm, shape, want_feasible)
        if slots is None:
            continue
        intent = intent_from_slots(shape, slots)
        validate(intent, gm.legend())
        start = _random_start(rng, gm)
        cost = reference_solve(gm, intent, start)
        feasible = not math.isinf(cost)
        if feasible != want_feasible:
            continue
        binding = feasible and _is_binding(gm, intent, start, cost)
        candidate = (gm, slots, intent, start, cost, binding)
        if want_binding and not binding and attempt < MAX_TRIES // 2:
            fallback = fallback or candidate
            continue
        break
    else:
        if fallback is None:
            raise RuntimeError(f"could not sample {ident} ({category}, {shape})")
        candidate = fallback
    gm, slots, intent, start, cost, binding = candidate
    if category == "soft" and alt_rng is not None:
        # Prefer an instance where the preference changes the route. A separate
        # random stream is used so that no other instruction changes.
        binding = _soft_detour(gm, intent, start)
        for attempt in range(MAX_TRIES if not binding else 0):
            gm2 = maps[(first_map + attempt) % len(maps)]
            slots2 = _choose_slots(alt_rng, gm2, shape, True)
            intent2 = intent_from_slots(shape, slots2)
            start2 = _random_start(alt_rng, gm2)
            cost2 = reference_solve(gm2, intent2, start2)
            if not math.isinf(cost2) and _soft_detour(gm2, intent2, start2):
                gm, slots, intent, start, cost, binding = gm2, slots2, intent2, start2, cost2, True
                break
    words = {k: display(v) for k, v in slots.items()}
    return Instruction(
        id=ident, category=category, source=source, map_id=gm.map_id, start=start,
        texts=tuple(w.format(**words) for w in wordings), gold_intent=intent,
        gold_ltl=gold_ltl(intent), feasible=not math.isinf(cost),
        ref_cost=None if math.isinf(cost) else round(float(cost), 9), binding=binding,
        checks=tuple(instruction_checks(intent)))


def generate(maps: list[GridMap], template_plan=TEMPLATE_PLAN, handwritten=HANDWRITTEN,
             seed: int = BENCHMARK_SEED) -> list[Instruction]:
    """Generate the benchmark deterministically, spreading instructions over maps."""
    rng = np.random.default_rng(seed)
    jobs = []  # (category, shape, wordings, source)
    for category, shape, count, family in template_plan:
        for _ in range(count):
            picks = rng.choice(len(TEMPLATES[family]), size=3, replace=False)
            jobs.append((category, shape, [TEMPLATES[family][i] for i in picks], "template"))
    for category, shape, wordings in handwritten:
        jobs.append((category, shape, wordings, "handwritten"))

    counters = {c: 0 for c in CATEGORIES}
    items = []
    for n, (category, shape, wordings, source) in enumerate(jobs):
        ident = f"{category}_{counters[category]:03d}"
        counters[category] += 1
        items.append(make_instruction(rng, maps, n % len(maps), category, shape,
                                      wordings, source, ident,
                                      alt_rng=np.random.default_rng([seed, n])))
    return items
