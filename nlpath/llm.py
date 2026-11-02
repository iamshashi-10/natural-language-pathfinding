"""The ONLY place that calls a language model (CLAUDE.md).

* ``GeminiClient`` — Google Gemini REST API via the standard library (no SDK),
  temperature 0, JSON output, client-side rate limiting, backoff on HTTP 429/5xx.
* ``CachedLLM`` — wraps any client; caches every response on disk under
  ``cache/llm/`` keyed by a hash of (model, prompt, params). The cache is
  committed, so every number can be reproduced without an API key; it is also
  the log of raw responses used in the error analysis.
* ``FakeLLM`` — a scripted client for tests and dry runs. Tests never touch the
  network.

The API key is read from the environment or ``.env`` (``GEMINI_API_KEY``) and
is never printed or logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "llm"

# ---- configuration: the one place the model is named (plan D2) ---------------
# Pinned exact version (not the "-latest" alias) so every call uses the same model.
# gemini-3.8-flash answered correctly but took 70-230 s per call on the free tier;
# gemini-3.5-flash-lite answered the same test prompts correctly in ~1.5 s (2026-11-03).
GEMINI_MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
REQUESTS_PER_MINUTE = 8       # stay under the free-tier per-minute cap
MAX_HTTP_ATTEMPTS = 6         # per call, for 429 / 5xx / network errors
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    """The model could not be reached or returned no usable text."""


class QuotaExhausted(LLMError):
    """Rate limit still hit after all retries (e.g. the daily free-tier cap)."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_s: float      # of the original call (cache hits report the stored value)
    cached: bool


class LLM(Protocol):
    model: str

    def complete(self, prompt: str) -> LLMResponse: ...


# ----- key ------------------------------------------------------------------


def load_api_key() -> str:
    """GEMINI_API_KEY from the environment, else from ``.env`` at the repo root."""
    key = os.environ.get("GEMINI_API_KEY")
    env_file = REPO / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "GEMINI_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set (environment or .env)")
    return key


# ----- Gemini ---------------------------------------------------------------


class GeminiClient:
    """Minimal Gemini client: one prompt in, one text (JSON) answer out."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None,
                 requests_per_minute: float = REQUESTS_PER_MINUTE,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model
        self._key = api_key  # loaded on the first real call, so cache-only runs need no key
        self._interval = 60.0 / requests_per_minute
        self._last_call = 0.0
        self._sleep = sleep

    def _wait_for_slot(self) -> None:
        """Client-side rate limiter: at most ``requests_per_minute`` calls."""
        wait = self._last_call + self._interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _post(self, body: dict) -> dict:
        if self._key is None:
            self._key = load_api_key()
        request = urllib.request.Request(
            API_URL.format(model=self.model), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key})
        with urllib.request.urlopen(request, timeout=300) as response:  # slow on free tier
            return json.loads(response.read())

    def complete(self, prompt: str) -> LLMResponse:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": TEMPERATURE,
                                     "responseMimeType": "application/json"}}
        for attempt in range(MAX_HTTP_ATTEMPTS):
            self._wait_for_slot()
            started = time.monotonic()
            try:
                data = self._post(body)
            except urllib.error.HTTPError as err:
                if err.code == 429 or err.code >= 500:
                    if attempt == MAX_HTTP_ATTEMPTS - 1:
                        if err.code == 429:
                            raise QuotaExhausted("rate limit / quota exhausted") from None
                        raise LLMError(f"HTTP {err.code} from Gemini") from None
                    self._sleep(min(120.0, 5.0 * 2 ** attempt))  # exponential backoff
                    continue
                raise LLMError(f"HTTP {err.code} from Gemini: {err.reason}") from None
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt == MAX_HTTP_ATTEMPTS - 1:
                    raise LLMError(f"network error: {err}") from None
                self._sleep(5.0 * 2 ** attempt)
                continue
            return LLMResponse(_extract_text(data), time.monotonic() - started, cached=False)
        raise LLMError("unreachable")


def _extract_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        # Blocked or empty answer: return it as text so the caller's validator
        # rejects it and the retry budget handles it like any other bad answer.
        return json.dumps({"error": "no candidates", "feedback": data.get("promptFeedback")})


# ----- cache ----------------------------------------------------------------


def cache_key(model: str, prompt: str, params: dict) -> str:
    blob = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedLLM:
    """Disk cache in front of any client; identical requests are answered from disk."""

    def __init__(self, client: LLM, cache_dir: Path = CACHE_DIR) -> None:
        self.client = client
        self.model = client.model
        self.cache_dir = Path(cache_dir)
        self.params = {"temperature": TEMPERATURE, "response": "json"}

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, prompt: str) -> LLMResponse:
        path = self._path(cache_key(self.model, prompt, self.params))
        if path.exists():
            entry = json.loads(path.read_text())
            return LLMResponse(entry["response"], entry["latency_s"], cached=True)
        response = self.client.complete(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"model": self.model, "params": self.params, "prompt": prompt,
                 "response": response.text, "latency_s": round(response.latency_s, 3),
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.write_text(json.dumps(entry, indent=1) + "\n")
        return LLMResponse(response.text, response.latency_s, cached=False)


# ----- fake -----------------------------------------------------------------


class FakeLLM:
    """Scripted client: ``respond(prompt) -> text``. Records every prompt it sees."""

    def __init__(self, respond: Callable[[str], str], model: str = "fake") -> None:
        self.model = model
        self._respond = respond
        self.prompts: list[str] = []

    @classmethod
    def from_list(cls, answers: list[str]) -> FakeLLM:
        """Answer with ``answers`` in order (one per call)."""
        queue = list(answers)
        return cls(lambda _prompt: queue.pop(0))

    def complete(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(self._respond(prompt), 0.0, cached=False)


def list_models(api_key: str | None = None) -> list[str]:
    """Names of the models this API key can call with generateContent."""
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": api_key or load_api_key()})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read())
    return sorted(m["name"].removeprefix("models/") for m in data.get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", []))


def default_llm() -> CachedLLM:
    """The configured model behind the on-disk cache — what every method uses."""
    return CachedLLM(GeminiClient())
