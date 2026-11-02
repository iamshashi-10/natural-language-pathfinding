"""Stage 1 (M3): instruction → intent form, via the LLM (plan §2.3).

The model receives ONLY the instruction and the list of region names — never
the grid, never a coordinate (CLAUDE.md, "the one rule"). Its answer is
accepted only if it parses as JSON, matches the intent-form schema, and is
well-formed for this map; otherwise the error is fed back (≤ 3 calls).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nlpath.grid import GridMap
from nlpath.intent import IntentError, IntentForm, validate
from nlpath.llm import LLM
from nlpath.pipeline import plan
from nlpath.astar import SearchResult
from nlpath.prompting import AskResult, Rejected, ask_with_retries, legend_text

PROMPT = """You turn a navigation instruction for an indoor robot into a JSON intent form.

{legend}

Return one JSON object with exactly these fields:
  "goal_region":       the region where the robot must end (a string)
  "forbidden_regions": regions that must never be entered (list)
  "ordered_waypoints": regions to visit before the goal, in the required order (list)
  "preferred_regions": regions to use when convenient, a soft preference (list)
  "avoided_regions":   regions to stay out of when convenient, a soft preference (list)

Rules:
- Use only region names from the list above, spelled exactly as listed.
- Something the robot must never do is forbidden; something it should do or
  avoid "if possible" / "preferably" is a soft preference.
- Do not list the goal among the waypoints. A region must not be both forbidden
  and visited, nor both preferred and avoided.

Example 1
Instruction: Go to the office, but stop at the atrium first and do not enter the lab.
Answer: {{"goal_region": "office", "forbidden_regions": ["lab"], "ordered_waypoints": ["atrium"], "preferred_regions": [], "avoided_regions": []}}

Example 2
Instruction: Head to the stairwell; use the corridor if you can and try to avoid the workshop.
Answer: {{"goal_region": "stairwell", "forbidden_regions": [], "ordered_waypoints": [], "preferred_regions": ["corridor"], "avoided_regions": ["workshop"]}}

Instruction: {instruction}
Answer:"""


def build_prompt(instruction: str, legend: list[str]) -> str:
    return PROMPT.format(legend=legend_text(legend), instruction=instruction)


def intent_validator(legend: list[str]):
    """Validator for ``ask_with_retries``: JSON → schema → well-formedness."""
    def check(text: str) -> IntentForm:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise Rejected(f"the answer is not valid JSON ({err.msg})") from None
        try:
            intent = IntentForm.from_dict(data)
            validate(intent, legend)
        except IntentError as err:
            raise Rejected(str(err)) from None
        return intent
    return check


def parse_instruction(llm: LLM, instruction: str, legend: list[str]) -> AskResult[IntentForm]:
    """Ask the LLM for an intent form; ``result.value`` is None if all calls failed."""
    return ask_with_retries(llm, build_prompt(instruction, legend), intent_validator(legend))


@dataclass
class M3Output:
    parse: AskResult[IntentForm]
    result: SearchResult | None     # None if parsing failed


def run_m3(llm: LLM, gm: GridMap, instruction: str, start) -> M3Output:
    """Our method: parse to an intent form, then plan with the verified pipeline."""
    parsed = parse_instruction(llm, instruction, gm.legend())
    if not parsed.ok:
        return M3Output(parsed, None)
    return M3Output(parsed, plan(gm, parsed.value, start))
