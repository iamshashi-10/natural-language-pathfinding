"""Tests for nlpath.llm, nlpath.prompting and nlpath.parser — no network."""

import json
import urllib.error

import pytest

from nlpath import llm as llm_mod
from nlpath.intent import IntentForm
from nlpath.llm import CachedLLM, FakeLLM, GeminiClient, QuotaExhausted, cache_key
from nlpath.parser import build_prompt, parse_instruction, run_m3
from nlpath.prompting import MAX_CALLS, legend_text
from reference import make_map

LEGEND = ["lab", "workshop", "store_room", "corridor"]
GOOD = json.dumps({"goal_region": "lab", "forbidden_regions": ["workshop"],
                   "ordered_waypoints": [], "preferred_regions": [], "avoided_regions": []})


# ----- the one rule ---------------------------------------------------------


def test_prompt_contains_instruction_and_legend_but_no_grid():
    gm = make_map(["#######", "#.....#", "#.###.#", "#.WWWL#", "#######"])
    fake = FakeLLM.from_list([GOOD])
    run_m3(fake, gm, "Go to the lab, not through the workshop.", (3, 1))
    (prompt,) = fake.prompts
    assert legend_text(gm.legend()) in prompt
    assert "Go to the lab, not through the workshop." in prompt
    assert "#" not in prompt and "(3, 1)" not in prompt and "[3, 1]" not in prompt


# ----- parser and retries ---------------------------------------------------


def test_parse_first_try():
    res = parse_instruction(FakeLLM.from_list([GOOD]), "…", LEGEND)
    assert res.ok and res.calls == 1
    assert res.value == IntentForm("lab", ("workshop",))


@pytest.mark.parametrize("bad, fragment", [
    ("not json", "not valid JSON"),
    ('{"goal_region": "kitchen"}', "unknown region 'kitchen'"),
    ('{"goal_region": "lab", "forbidden_regions": ["lab"]}', "also forbidden"),
    ('{"goal": "lab"}', "unknown field"),
])
def test_retry_feeds_back_the_error(bad, fragment):
    fake = FakeLLM.from_list([bad, GOOD])
    res = parse_instruction(fake, "go to the lab", LEGEND)
    assert res.ok and res.calls == 2
    assert fragment in res.errors[0]
    assert fragment in fake.prompts[1] and bad in fake.prompts[1]


def test_retry_budget_is_three_calls():
    fake = FakeLLM.from_list(["nope"] * 5)
    res = parse_instruction(fake, "go to the lab", LEGEND)
    assert not res.ok and res.calls == MAX_CALLS == 3 and len(fake.prompts) == 3


def test_m3_end_to_end_with_parse_failure():
    gm = make_map(["#####", "#.L.#", "#####"])
    out = run_m3(FakeLLM.from_list(["x", "y", "z"]), gm, "lab", (1, 1))
    assert out.result is None and out.parse.calls == 3
    out = run_m3(FakeLLM.from_list(['{"goal_region": "lab"}']), gm, "lab", (1, 1))
    assert out.result.found and out.result.path == [(1, 1), (1, 2)]


# ----- cache ----------------------------------------------------------------


def test_cache_hits_do_not_call_the_model(tmp_path):
    fake = FakeLLM(lambda p: p.upper())
    cached = CachedLLM(fake, cache_dir=tmp_path)
    first = cached.complete("hello")
    second = cached.complete("hello")
    assert first.text == second.text == "HELLO"
    assert not first.cached and second.cached
    assert len(fake.prompts) == 1
    key = cache_key("fake", "hello", cached.params)
    entry = json.loads((tmp_path / key[:2] / f"{key}.json").read_text())
    assert entry["prompt"] == "hello" and entry["response"] == "HELLO"


def test_cache_key_depends_on_model_and_prompt():
    assert cache_key("a", "p", {}) != cache_key("b", "p", {})
    assert cache_key("a", "p", {}) != cache_key("a", "q", {})
    assert cache_key("a", "p", {"t": 0}) == cache_key("a", "p", {"t": 0})


# ----- Gemini client (network replaced) -------------------------------------


def gemini(monkeypatch, replies):
    """GeminiClient whose HTTP POST returns/raises ``replies`` in order."""
    sleeps = []
    client = GeminiClient(api_key="test-key", requests_per_minute=60, sleep=sleeps.append)
    queue = list(replies)

    def fake_post(body):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(client, "_post", fake_post)
    return client, sleeps


def ok_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def http_error(code):
    return urllib.error.HTTPError("url", code, "err", {}, None)


def test_gemini_backs_off_on_429(monkeypatch):
    client, sleeps = gemini(monkeypatch, [http_error(429), http_error(503), ok_reply("{}")])
    assert client.complete("p").text == "{}"
    assert [s for s in sleeps if s >= 5] == [5.0, 10.0]  # exponential backoff


def test_rate_limiter_spaces_concurrent_calls(monkeypatch):
    """Slots are reserved under a lock: 4 threads get starts 1 s apart (60 RPM)."""
    import threading
    clock = [100.0]
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock[0])
    waits = []
    client = GeminiClient(api_key="k", requests_per_minute=60, sleep=waits.append)
    threads = [threading.Thread(target=client._wait_for_slot) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(waits) == [1.0, 2.0, 3.0]  # the first call starts at once


def test_gemini_gives_up_with_quota_error(monkeypatch):
    client, _ = gemini(monkeypatch, [http_error(429)] * llm_mod.MAX_HTTP_ATTEMPTS)
    with pytest.raises(QuotaExhausted):
        client.complete("p")


def test_gemini_blocked_answer_becomes_rejectable_text(monkeypatch):
    client, _ = gemini(monkeypatch, [{"promptFeedback": {"blockReason": "OTHER"}}])
    assert "no candidates" in client.complete("p").text


def test_missing_key_is_reported_without_value(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod, "REPO", tmp_path)
    with pytest.raises(llm_mod.LLMError, match="GEMINI_API_KEY is not set"):
        llm_mod.load_api_key()


def test_key_read_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod, "REPO", tmp_path)
    (tmp_path / ".env").write_text('OTHER=1\nGEMINI_API_KEY="abc"\n')
    assert llm_mod.load_api_key() == "abc"


def test_prompt_builder_escapes_braces():
    assert '"goal_region": "office"' in build_prompt("x", LEGEND)
