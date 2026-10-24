"""The intent form: what the parser (LLM) returns and the compiler consumes.

Plan §2.3 and §3.1. An intent form names regions only — never coordinates —
so it is the single channel through which LLM output reaches the planner
(CLAUDE.md, "the one rule").

``validate`` enforces well-formedness (plan §3.1), the precondition of T1–T6.
Its error messages are written to be fed back to the LLM on a retry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

LIST_FIELDS = ("forbidden_regions", "ordered_waypoints",
               "preferred_regions", "avoided_regions")
FIELDS = ("goal_region", *LIST_FIELDS)


class IntentError(ValueError):
    """An intent form is malformed or not well-formed."""


@dataclass(frozen=True)
class IntentForm:
    """I = (g, X, (w_1..w_m), P, A) in plan §3.1."""

    goal_region: str
    forbidden_regions: tuple[str, ...] = ()
    ordered_waypoints: tuple[str, ...] = ()
    preferred_regions: tuple[str, ...] = ()
    avoided_regions: tuple[str, ...] = ()

    @property
    def sequence(self) -> tuple[str, ...]:
        """(u_1, …, u_{m+1}) = (w_1, …, w_m, g): the regions to enter, in order."""
        return (*self.ordered_waypoints, self.goal_region)

    def hard_part(self) -> IntentForm:
        """I_hard: the same form without soft preferences (plan §3.1)."""
        return IntentForm(self.goal_region, self.forbidden_regions,
                          self.ordered_waypoints)

    def to_dict(self) -> dict:
        return {"goal_region": self.goal_region,
                **{f: list(getattr(self, f)) for f in LIST_FIELDS}}

    @classmethod
    def from_dict(cls, data: object) -> IntentForm:
        """Build from parsed JSON, checking the schema (not the semantics).

        Missing list fields default to empty; unknown fields are rejected.
        """
        if not isinstance(data, dict):
            raise IntentError("the intent form must be a JSON object")
        unknown = sorted(set(data) - set(FIELDS))
        if unknown:
            raise IntentError(f"unknown field(s) {unknown}; allowed fields are {list(FIELDS)}")
        goal = data.get("goal_region")
        if not isinstance(goal, str) or not goal:
            raise IntentError("goal_region must be a non-empty string")
        lists = {}
        for f in LIST_FIELDS:
            value = data.get(f, [])
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise IntentError(f"{f} must be a list of region-name strings")
            lists[f] = tuple(value)
        return cls(goal, **lists)


def validate(intent: IntentForm, legend: Iterable[str]) -> None:
    """Raise ``IntentError`` unless ``intent`` is well-formed for this legend.

    Well-formed (plan §3.1):
      1. every name is in the legend (catches hallucinated rooms);
      2. the sequence (waypoints…, goal) has pairwise distinct elements;
      3. no sequence element is forbidden;
      4. no region is both preferred and avoided.
    """
    legend = list(legend)
    for f in FIELDS:
        names = [intent.goal_region] if f == "goal_region" else getattr(intent, f)
        for name in names:
            if name not in legend:
                raise IntentError(f"{f}: unknown region {name!r}; "
                                  f"choose from {legend}")
    seq = intent.sequence
    repeated = sorted({n for n in seq if seq.count(n) > 1})
    if repeated:
        raise IntentError(f"region(s) {repeated} appear more than once among "
                          "ordered_waypoints and goal_region")
    clash = [n for n in seq if n in intent.forbidden_regions]
    if clash:
        raise IntentError(f"region(s) {clash} must be visited but are also forbidden")
    both = sorted(set(intent.preferred_regions) & set(intent.avoided_regions))
    if both:
        raise IntentError(f"region(s) {both} are both preferred and avoided")
