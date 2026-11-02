"""Check the Gemini setup: key found, which Flash models it can use, one test call.

Usage:  python scripts/check_gemini.py

The test call is not cached. The key itself is never printed.
"""

from __future__ import annotations

from nlpath.llm import GEMINI_MODEL, GeminiClient, LLMError, list_models


def main() -> None:
    try:
        models = list_models()
    except LLMError as err:
        raise SystemExit(f"setup problem: {err}") from None
    flash = [m for m in models if "flash" in m]
    print("Flash models available to this key:")
    for m in flash:
        print("  ", m, "  <- configured" if m == GEMINI_MODEL else "")
    if GEMINI_MODEL not in models:
        raise SystemExit(f"configured model {GEMINI_MODEL!r} is NOT available; "
                         "change GEMINI_MODEL in nlpath/llm.py")
    reply = GeminiClient().complete('Answer with the JSON object {"ok": true}.')
    print(f"test call to {GEMINI_MODEL}: {reply.text.strip()} ({reply.latency_s:.1f}s)")


if __name__ == "__main__":
    main()
