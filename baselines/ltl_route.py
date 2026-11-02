"""M2: NL → LTLf → automaton → product planner (the formal-specification route).

Plan §4.2, alternative A. The LLM translates the instruction into a finite-trace
LTL formula over region names; Spot checks the syntax, turns the formula into a
finite automaton (LTLf → LTL via ``from_ltlf`` → deterministic automaton →
``to_finite``), and a shortest-path search runs on the product
(grid cell × automaton state) with the same step costs as our planner.

The trace of a path is the sequence of region labels of its cells; ``last``
(true only at the final position) is written ``last`` and replaced by
``!X[!]1`` before Spot reads the formula.

Variants (plan §4.2): M2 (LLM formula), M2-oracle (gold formula from T6 — also
"M2-drop", since the gold formula has no soft preferences) and M2-harden
(soft preferences turned into hard constraints).

Spot is an optional dependency (conda-forge); it is imported only when used.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from nlpath.astar import SearchResult, astar, grid_moves
from nlpath.grid import Cell, GridMap
from nlpath.intent import IntentForm
from nlpath.llm import LLM
from nlpath.prompting import AskResult, Rejected, ask_with_retries, legend_text

LAST = "(!X[!]1)"
_LAST_WORD = re.compile(r"\blast\b")
KEYWORDS = {"last"}

PROMPT = """You translate a navigation instruction for an indoor robot into a finite-trace
linear temporal logic (LTLf) formula.

{legend}

The robot's path is a finite sequence of positions. At each position exactly one
region name is true (the region the robot is in), or none (outside every region).
Write the formula with:
  region names as atomic propositions (spelled exactly as listed),
  ! (not), & (and), | (or), -> (implies), parentheses,
  F p (eventually p), G p (always p), p U q (p until q, and q eventually happens),
  X p (next), and the keyword last, which is true only at the final position.

Return one JSON object: {{"ltl": "<formula>"}}

Example 1
Instruction: Go to the office, but stop at the atrium first and do not enter the lab.
Answer: {{"ltl": "G !lab & (!office U (atrium & F(office & last)))"}}

Example 2
Instruction: Visit the print room, then the store room, and end in the stairwell.
Answer: {{"ltl": "!(store_room | stairwell) U (print_room & (!stairwell U (store_room & F(stairwell & last))))"}}

Instruction: {instruction}
Answer:"""


def _spot():
    import spot  # optional dependency: conda install -c conda-forge spot
    return spot


def with_last(text: str) -> str:
    """Replace the keyword ``last`` by its LTLf meaning, "there is no next position"."""
    return _LAST_WORD.sub(LAST, text)


def build_prompt(instruction: str, legend: list[str]) -> str:
    return PROMPT.format(legend=legend_text(legend), instruction=instruction)


def parse_ltl(text: str, legend: list[str]):
    """Parse an LTLf formula string; raise ``Rejected`` with a readable reason."""
    spot = _spot()
    try:
        # Check the syntax as the model wrote it (``last`` parses as an atom),
        # so the diagnostic shown to the model matches its own text.
        spot.formula(text)
    except spot.parse_error as err:
        # Spot's diagnostic (with a caret under the problem) is fed back as is.
        raise Rejected(f"the formula is not valid LTL syntax:\n{str(err).strip()}") from None
    formula = spot.formula(with_last(text))
    unknown = sorted({str(ap) for ap in spot.atomic_prop_collect(formula)}
                     - set(legend) - KEYWORDS)
    if unknown:
        raise Rejected(f"unknown region name(s) {unknown}; choose from {legend}")
    return formula


def ltl_validator(legend: list[str]):
    """Validator for ``ask_with_retries``: JSON → {"ltl": str} → Spot syntax check."""
    def check(text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise Rejected(f"the answer is not valid JSON ({err.msg})") from None
        if not isinstance(data, dict) or not isinstance(data.get("ltl"), str):
            raise Rejected('answer with a JSON object {"ltl": "<formula>"}')
        parse_ltl(data["ltl"], legend)
        return data["ltl"]
    return check


def translate(llm: LLM, instruction: str, legend: list[str]) -> AskResult[str]:
    return ask_with_retries(llm, build_prompt(instruction, legend), ltl_validator(legend))


def ltl_equivalent(a: str, b: str) -> bool:
    """Are two LTLf formulas equivalent (on finite traces)? For translation accuracy."""
    spot = _spot()
    fa, fb = (spot.from_ltlf(spot.formula(with_last(x))) for x in (a, b))
    return bool(spot.are_equivalent(fa, fb))


# ----- automaton ------------------------------------------------------------


@dataclass(frozen=True)
class FiniteAutomaton:
    """Deterministic finite automaton over map labels.

    ``delta[q][j]`` is the successor of state q on reading a cell with label
    class j (j = index into ``classes``; the last class is "no region"), or -1.
    """

    init: int
    delta: np.ndarray       # int (states, classes)
    accepting: np.ndarray   # bool (states,)
    classes: tuple[str | None, ...]


def build_automaton(formula_text: str, legend: list[str]) -> FiniteAutomaton:
    """LTLf formula → deterministic finite automaton over the map's label classes."""
    spot = _spot()
    import buddy

    formula = parse_ltl(formula_text, legend)
    aut = spot.to_finite(spot.translate(spot.from_ltlf(formula), "buchi",
                                        "deterministic", "state-based"))
    bdd_dict = aut.get_dict()
    aps = [str(ap) for ap in aut.ap()]
    classes = (*legend, None)

    def letter(label: str | None):
        b = buddy.bddtrue
        for ap in aps:
            var = buddy.bdd_ithvar(bdd_dict.varnum(spot.formula(ap)))
            b = buddy.bdd_and(b, var if ap == label else buddy.bdd_not(var))
        return b

    letters = [letter(c) for c in classes]
    n = aut.num_states()
    delta = np.full((n, len(classes)), -1, dtype=int)
    for q in range(n):
        for edge in aut.out(q):
            for j, b in enumerate(letters):
                if buddy.bdd_and(edge.cond, b) != buddy.bddfalse:
                    delta[q, j] = edge.dst   # deterministic: at most one edge matches
    accepting = np.array([bool(aut.state_is_accepting(q)) for q in range(n)])
    return FiniteAutomaton(aut.get_init_state_number(), delta, accepting, classes)


# ----- product planner ------------------------------------------------------


def plan_ltl(gm: GridMap, formula_text: str, start: Cell,
             weights: np.ndarray | None = None) -> SearchResult:
    """Cheapest physically valid path from ``start`` whose label trace satisfies the
    formula. Uniform-cost search (A* with h = 0) on cells × automaton states.

    Step cost is ``d(v,u) * weights[u]`` (default: all 1, i.e. path length);
    moves are physical moves with no corner cutting.
    """
    dfa = build_automaton(formula_text, gm.legend())
    class_of = np.full(gm.shape, len(dfa.classes) - 1, dtype=int)
    for j, name in enumerate(dfa.classes[:-1]):
        class_of[gm.region_mask(name)] = j
    if weights is None:
        weights = np.ones(gm.shape)
    start = (int(start[0]), int(start[1]))
    if not gm.free[start]:
        return SearchResult(None, math.inf, 0, "start cell is a wall")
    q0 = int(dfa.delta[dfa.init, class_of[start]])
    if q0 < 0:
        return SearchResult(None, math.inf, 0, "the start cell already violates the formula")

    def successors(state) -> Iterator:
        v, q = state
        for u, step in grid_moves(gm.free, v):
            q2 = int(dfa.delta[q, class_of[u]])
            if q2 >= 0:
                yield (u, q2), step * float(weights[u])

    result = astar((start, q0), lambda s: bool(dfa.accepting[s[1]]), successors,
                   lambda s: 0.0)
    if not result.found:
        return SearchResult(None, math.inf, result.nodes_expanded,
                            "no path satisfies the formula")
    return SearchResult([v for v, _ in result.path], result.cost, result.nodes_expanded)


def harden(gold_intent: IntentForm, gold_formula: str) -> str:
    """M2-harden (plan §4.2): "avoid if you can" becomes "never enter" (``G !a``).

    A preferred region is left unconstrained — the path uses it or not, whichever
    is shortest. Hardening it (``G corridor``) is unsatisfiable whenever the start
    or goal lies outside the corridor, i.e. on every benchmark instruction.
    """
    if not gold_intent.avoided_regions:
        return gold_formula
    avoided = " & ".join(f"G !{a}" for a in gold_intent.avoided_regions)
    return f"({gold_formula}) & {avoided}"


@dataclass
class M2Output:
    translation: AskResult[str]
    result: SearchResult | None     # None if no valid formula was obtained


def run_m2(llm: LLM, gm: GridMap, instruction: str, start: Cell) -> M2Output:
    """M2: LLM → formula (≤ 3 calls) → automaton product planner."""
    translation = translate(llm, instruction, gm.legend())
    if not translation.ok:
        return M2Output(translation, None)
    return M2Output(translation, plan_ltl(gm, translation.value, start))
