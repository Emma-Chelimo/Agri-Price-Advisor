"""
Tests for the language-model assistant (Phase 7). No real API is called.

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import assistant as A  # noqa: E402
from src import decision as D  # noqa: E402
from tests.test_decision import make_fc  # noqa: E402


@pytest.fixture
def payload():
    fc = make_fc([1000] * 3, sigma=0.1, p0=1000.0)
    costs = D.Costs(20, 0.01, 3.0)
    adv = D.advise(fc, 500, costs)
    return A.build_payload(crop="Maize", market="Home", quantity_kg=500, fc=fc, adv=adv,
                           costs=costs, months_old=0)


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, messages, max_tokens=400):
        self.calls.append((system, messages))
        if self.error:
            raise A.LLMError(self.error)
        return self.reply


def test_payload_is_json_and_has_the_key_figures(payload):
    s = json.dumps(payload)
    assert payload["recommendation"]["action"] == "Sell now"
    assert payload["recommendation"]["earnings_if_sold_now_ugx"] == 500_000
    assert len(payload["options"]) == 4 and len(payload["forecast"]["months"]) == 3
    assert "not_covered_by_this_system" in payload and "NaN" not in s


def test_grounding_allows_rounding_and_rejects_invented_numbers(payload):
    assert A.check_grounding("Sell now for about 500,000 UGX (January 2026).", payload).ok
    assert A.check_grounding("Sell now for about 501,000 UGX.", payload).ok        # 0.2% off
    bad = A.check_grounding("Prices will reach 1,400 UGX/kg next month.", payload)
    assert not bad.ok and bad.unmatched == [1400.0]
    assert A.check_grounding("Store for 3 months or 2 months.", payload).ok          # small counts


def test_explain_uses_data_block_and_rules(payload):
    llm = FakeLLM("Sell your 500 kg now for about 500,000 UGX.")
    r = A.explain(llm, payload)
    assert r.ok and "500,000" in r.text
    system, messages = llm.calls[0]
    assert "ONLY the facts" in system and "DATA:" in messages[0]["content"]


def test_explain_falls_back_when_numbers_are_invented(payload):
    r = A.explain(FakeLLM("You could earn 750,000 UGX by waiting."), payload)
    assert not r.ok and 750000.0 in r.unmatched_numbers


def test_explain_falls_back_on_errors_and_without_a_model(payload):
    assert not A.explain(FakeLLM(error="HTTP 401"), payload).ok
    r = A.explain(None, payload)
    assert not r.ok and "no language model" in r.reason


def test_answer_keeps_history_and_refusals_pass(payload):
    llm = FakeLLM("I don't have that information. I can explain the storage options.")
    history = [{"role": "user", "content": "Why sell now?"},
               {"role": "assistant", "content": "Because storing costs more than it is likely to earn."}]
    r = A.answer(llm, payload, "Will it rain next week?", history)
    assert r.ok and r.text.startswith("I don't have")
    msgs = llm.calls[0][1]
    assert msgs[-1]["content"] == "Will it rain next week?" and msgs[-3]["content"] == "Why sell now?"


def test_answer_with_invented_number_returns_refusal(payload):
    r = A.answer(FakeLLM("A buyer in Kampala pays 1,900 UGX/kg."), payload, "Who pays more?")
    assert not r.ok and r.text == A.REFUSAL


def test_get_llm_configuration(monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AGRIPRICE_LLM_PROVIDER",
              "AGRIPRICE_LLM_MODEL", "OPENAI_BASE_URL"]:
        monkeypatch.delenv(k, raising=False)
    assert A.get_llm({}) is None
    llm = A.get_llm({"ANTHROPIC_API_KEY": "k"})
    assert isinstance(llm, A.AnthropicLLM) and llm.model == A.DEFAULT_ANTHROPIC_MODEL
    assert A.get_llm({"OPENAI_API_KEY": "k"}) is None                 # model is required
    llm = A.get_llm({"OPENAI_API_KEY": "k", "AGRIPRICE_LLM_MODEL": "m",
                     "OPENAI_BASE_URL": "http://localhost:11434/v1"})
    assert isinstance(llm, A.OpenAICompatibleLLM) and llm.base_url.endswith("11434/v1")
    assert A.get_llm({"ANTHROPIC_API_KEY": "k", "AGRIPRICE_LLM_PROVIDER": "none"}) is None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_anthropic_request_format(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["headers"] = req.full_url, dict(req.header_items())
        seen["body"] = json.loads(req.data)
        return _Resp(json.dumps({"content": [{"type": "text", "text": "Hello"}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = A.AnthropicLLM(api_key="secret").complete("SYS", [{"role": "user", "content": "hi"}])
    assert out == "Hello" and seen["url"] == A.ANTHROPIC_URL
    assert seen["headers"]["X-api-key"] == "secret"
    assert seen["headers"]["Anthropic-version"] == A.ANTHROPIC_VERSION
    assert seen["body"]["system"] == "SYS" and seen["body"]["model"] == A.DEFAULT_ANTHROPIC_MODEL


def test_openai_request_format(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        seen["auth"] = dict(req.header_items())["Authorization"]
        return _Resp(json.dumps({"choices": [{"message": {"content": "Hi"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = A.OpenAICompatibleLLM("k", "m", "http://x/v1/").complete("SYS", [{"role": "user", "content": "q"}])
    assert out == "Hi" and seen["url"] == "http://x/v1/chat/completions"
    assert seen["auth"] == "Bearer k" and seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}


def test_grounding_is_unit_aware(payload):
    # one month of storage costs 10,000 UGX in this payload; a price per kg of
    # 9,999 must not match it just because the numbers are close
    assert A.check_grounding("Storing one month costs 10,000 UGX.", payload).ok
    assert not A.check_grounding("A trader will pay 9,999 UGX/kg.", payload).ok
    assert not A.check_grounding("The chance is 500,000%.", payload).ok


def test_grounding_rejects_markets_not_in_payload(payload):
    known = ["Home", "Gulu", "Owino"]
    assert A.check_grounding("Selling at Home is best.", payload, known_markets=known).ok
    r = A.check_grounding("Try Gulu instead.", payload, known_markets=known)
    assert not r.ok and r.unknown_markets == ["Gulu"]
