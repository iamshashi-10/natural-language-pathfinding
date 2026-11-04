"""Error analysis: attribute every failure to a stage and a type (plan §4.6, Day 13).

Each result row of scripts/run_all.py gets one label — the first rule that
applies, in the order listed for its method. Stages:

* ``interpretation`` — the LLM's reading of the sentence was wrong;
* ``planning``       — the path is wrong although the reading was right
                       (for M3 this would contradict T2–T4, i.e. a bug);
* ``path``           — M1 only: the path the model wrote is unusable;
* ``ok``             — correct (including a correct NO_PATH).
"""

from __future__ import annotations

OK = ("ok", "correct")


def _intent_diff(parsed: dict, gold: dict) -> tuple[str, str]:
    """Why a parsed intent form differs from the gold one (M3)."""
    if parsed["goal_region"] != gold["goal_region"]:
        return "interpretation", "wrong goal region"
    pw, gw = parsed["ordered_waypoints"], gold["ordered_waypoints"]
    if pw != gw:
        if sorted(pw) == sorted(gw):
            return "interpretation", "ordering inverted"
        if set(gw) - set(pw):
            return "interpretation", "waypoint / ordering dropped"
        return "interpretation", "extra waypoint"
    pf, gf = set(parsed["forbidden_regions"]), set(gold["forbidden_regions"])
    pa, ga = set(parsed["avoided_regions"]), set(gold["avoided_regions"])
    pp, gp = set(parsed["preferred_regions"]), set(gold["preferred_regions"])
    if (ga & pf) or (gp & pf):
        return "interpretation", "soft preference read as prohibition"
    if gf & pa:
        return "interpretation", "prohibition read as soft preference"
    if gf - pf:
        return "interpretation", "prohibition missed"
    if (gp - pp) or (ga - pa):
        return "interpretation", "soft preference missed"
    return "interpretation", "extra constraint"


def classify_m3(row: dict, gold_intent: dict) -> tuple[str, str]:
    if row["correct"]:
        if row.get("parse_exact", True):
            return OK
        return "ok", "correct despite imperfect parse"
    if row.get("parsed") is None:
        return "interpretation", "no valid intent form after 3 calls"
    if not row.get("parse_exact", False):
        return _intent_diff(row["parsed"], gold_intent)
    return "planning", "BUG: correct parse but wrong result"


def classify_m2(row: dict, gold_intent: dict) -> tuple[str, str]:
    if row["correct"]:
        return OK if row.get("ltl_equivalent") else ("ok", "correct despite non-equivalent formula")
    if row.get("formula") is None:
        return "interpretation", "no syntactically valid formula after 3 calls"
    if not row.get("ltl_equivalent"):
        formula = row["formula"]
        if any(f"G !{a}" in formula.replace("G!", "G !") or f"G !({a}" in formula
               for a in gold_intent["avoided_regions"] + gold_intent["preferred_regions"]):
            return "interpretation", "soft preference read as prohibition"
        return "interpretation", "semantically wrong formula"
    return "planning", "BUG: equivalent formula but wrong result"


def classify_m1(row: dict, feasible: bool) -> tuple[str, str]:
    if row["correct"]:
        return OK
    if row.get("format_failure"):
        return "interpretation", "no well-formed answer after 3 calls"
    if row["path"] is None:
        return "interpretation", "said impossible on a feasible instruction"
    if not feasible:
        return "path", "invented a path on an infeasible instruction"
    problems = " ".join(row["problems"])
    for fragment, label in (("outside the map", "path leaves the map"),
                            ("path starts at", "wrong start cell"),
                            ("not a move to a neighbouring", "discontinuity (jump)"),
                            ("not an allowed cell", "walks through a wall or forbidden cell"),
                            ("cuts a corner", "cuts a corner"),
                            ("ends_in", "stops short of the goal")):
        if fragment in problems:
            return "path", label
    return "interpretation", "walkable path violating a constraint"


def classify_planner_only(row: dict) -> tuple[str, str]:
    """M0 / oracle methods: no LLM, so every failure is in planning (by design for M0)."""
    if row["correct"]:
        return OK
    if row["returned_path"]:
        return "planning", "path ignores a constraint" if row["valid"] else "invalid path"
    return "planning", "NO_PATH on a feasible instruction"


def classify(row: dict, gold_intent: dict, feasible: bool) -> tuple[str, str]:
    method = row["method"]
    if method == "M3":
        return classify_m3(row, gold_intent)
    if method == "M2":
        return classify_m2(row, gold_intent)
    if method == "M1":
        return classify_m1(row, feasible)
    return classify_planner_only(row)
