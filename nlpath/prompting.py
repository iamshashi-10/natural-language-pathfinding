"""What M1, M2 and M3 share when they ask the LLM (plan §4.2 fairness rules).

* ``legend_text`` — the same region legend text for every method.
* ``ask_with_retries`` — the same retry budget: up to ``MAX_CALLS`` calls, each
  retry shown the previous answer and the validator's error message.

Each method supplies only its own prompt and its own validator.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from nlpath.llm import LLM

MAX_CALLS = 3
T = TypeVar("T")


def legend_text(legend: list[str]) -> str:
    """The region list shown to the model, identical across methods."""
    return "Regions on this map: " + ", ".join(legend)


class Rejected(ValueError):
    """A validator's verdict on a model answer; the message is fed back to the model."""


@dataclass
class AskResult(Generic[T]):
    value: T | None                 # validated answer, or None if every call failed
    calls: int
    latency_s: float                # sum over calls (original latencies for cache hits)
    raw: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


def retry_prompt(prompt: str, previous: str, error: str) -> str:
    return (f"{prompt}\n\nYour previous answer was:\n{previous}\n\n"
            f"It was rejected: {error}\nPlease answer again, fixing this problem.")


def ask_with_retries(llm: LLM, prompt: str, validate: Callable[[str], T],
                     max_calls: int = MAX_CALLS) -> AskResult[T]:
    """Ask, validate, and retry with error feedback (≤ ``max_calls`` calls).

    ``validate(text)`` returns the parsed value or raises ``Rejected``.
    """
    result: AskResult[T] = AskResult(None, 0, 0.0)
    current = prompt
    for _ in range(max_calls):
        started = time.monotonic()
        response = llm.complete(current)
        result.calls += 1
        result.latency_s += response.latency_s if response.cached else \
            time.monotonic() - started
        result.raw.append(response.text)
        try:
            result.value = validate(response.text)
            return result
        except Rejected as err:
            result.errors.append(str(err))
            current = retry_prompt(prompt, response.text, str(err))
    return result
