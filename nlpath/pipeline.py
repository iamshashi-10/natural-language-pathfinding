"""Planning pipeline from intent form to verified path (stages 2–3 + verifier).

    intent form ──validate/compile──▶ CompiledProblem ──planner──▶ path ──verify_path──▶

This is the single entry point for planning: every path it returns has passed
``verify_path`` (plan §3.5). A verification failure raises — it is a bug and is
never returned quietly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from nlpath.astar import SearchResult
from nlpath.compiler import CompiledProblem, compile_intent
from nlpath.grid import Cell, GridMap
from nlpath.intent import IntentForm
from nlpath.product import plan_product
from nlpath.verify import PathVerificationError, path_cost, verify_path

Planner = Callable[[CompiledProblem, Cell], SearchResult]


def plan(gm: GridMap, intent: IntentForm, start: Cell,
         planner: Planner = plan_product) -> SearchResult:
    """Plan an I-satisfying path from ``start``, or return NO_PATH with a reason.

    ``planner`` defaults to the product-graph search (M3); the M3-chain ablation
    passes ``baselines.chaining.plan_chain`` so its output is verified the same way.

    Raises:
        IntentError: the intent form is not well-formed.
        PathVerificationError: the planner returned an invalid path, or a cost
            that does not match the path (both are bugs).
    """
    problem = compile_intent(gm, intent)
    result = planner(problem, start)
    if not result.found:
        return result
    verify_path(result.path, gm, intent, start)
    recomputed = path_cost(result.path, gm, intent)
    if not math.isclose(recomputed, result.cost, rel_tol=1e-9, abs_tol=1e-9):
        raise PathVerificationError(
            f"planner reported cost {result.cost}, but the path costs {recomputed}")
    return result
