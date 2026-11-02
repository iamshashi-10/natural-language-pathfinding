"""M1: end-to-end LLM — the model sees the ASCII map and outputs the path itself.

Plan §4.2, alternative B. This baseline deliberately breaks "the one rule"
(CLAUDE.md): the LLM sees the grid and emits every coordinate. That is the
point of the comparison, and why it lives in ``baselines/``.

The validator only checks the answer's format (JSON, a list of [row, col]
integer pairs, or null for "impossible"); walkability is NOT fed back — that
would turn the baseline into a search procedure. The same retry budget as M2
and M3 applies to format errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nlpath.grid import REGION_LETTERS, Cell, GridMap
from nlpath.llm import LLM
from nlpath.prompting import AskResult, Rejected, ask_with_retries, legend_text

PROMPT = """You plan a path for an indoor robot on a grid map, following an instruction.

{legend}
Map key: '#' wall, '.' free cell outside every region, {letters}.
Cells are [row, col], both counted from 0; row 0 is the top line, col 0 the
leftmost character.

Map ({height} rows x {width} columns):
{grid}

Moves: from a cell the robot may step to any of its 8 neighbours that is not a
wall. A diagonal step is allowed only if both cells beside it are not walls.
The path must start at the start cell, follow the instruction, and end inside
the goal region.

Return one JSON object: {{"path": [[row, col], [row, col], ...]}}, listing every
cell of the path in order, starting with the start cell.
If the instruction cannot be followed on this map, return {{"path": null}}.

Example 1
Map (3 rows x 5 columns):
#####
#..L#
#####
Start: [1, 1]
Instruction: Go to the lab.
Answer: {{"path": [[1, 1], [1, 2], [1, 3]]}}

Example 2
Map (4 rows x 5 columns):
#####
#.W.#
#..L#
#####
Start: [1, 1]
Instruction: Go to the lab without entering the workshop.
Answer: {{"path": [[1, 1], [2, 1], [2, 2], [2, 3]]}}

Map ({height} rows x {width} columns):
{grid}
Start: [{r}, {c}]
Instruction: {instruction}
Answer:"""


def build_prompt(gm: GridMap, instruction: str, start: Cell) -> str:
    letters = ", ".join(f"'{REGION_LETTERS[n]}' {n}" for n in gm.legend())
    h, w = gm.shape
    return PROMPT.format(legend=legend_text(gm.legend()), letters=letters,
                         height=h, width=w, grid="\n".join(gm.to_rows()),
                         r=start[0], c=start[1], instruction=instruction)


def path_validator(text: str) -> list[Cell] | str:
    """Format check only. Returns the path, or the string "NO_PATH" for null."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise Rejected(f"the answer is not valid JSON ({err.msg})") from None
    if not isinstance(data, dict) or "path" not in data:
        raise Rejected('answer with a JSON object {"path": [[row, col], ...]}')
    path = data["path"]
    if path is None:
        return "NO_PATH"
    if (not isinstance(path, list) or not path
            or not all(isinstance(p, list) and len(p) == 2
                       and all(isinstance(x, int) and not isinstance(x, bool) for x in p)
                       for p in path)):
        raise Rejected('"path" must be a non-empty list of [row, col] integer pairs, or null')
    return [(p[0], p[1]) for p in path]


@dataclass
class M1Output:
    ask: AskResult
    path: list[Cell] | None      # None: model said impossible, or no valid answer


def run_m1(llm: LLM, gm: GridMap, instruction: str, start: Cell) -> M1Output:
    ask = ask_with_retries(llm, build_prompt(gm, instruction, start), path_validator)
    path = ask.value if isinstance(ask.value, list) else None
    return M1Output(ask, path)
