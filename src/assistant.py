"""
Language-model assistant (Phase 7).

The assistant never computes anything. The app's own modules produce every
figure; this file packs those figures into a JSON "payload", asks a language
model to put them into plain words, and checks the answer before showing it.

    build_payload()    everything the dashboard computed, as plain JSON
    explain()          a short written explanation of the advice
    answer()           an answer to a farmer's question about the advice
    check_grounding()  rejects any reply that contains a number not found in
                       the payload (the model may round, but not invent)

If no model is configured, the call fails, or the check fails, the caller
gets `ok=False` and shows the template explanation instead.

Providers
---------
Configured with environment variables or Streamlit secrets:

    AGRIPRICE_LLM_PROVIDER   anthropic | openai | none
                             (default: anthropic if ANTHROPIC_API_KEY is set,
                             else openai if OPENAI_API_KEY is set, else none)
    ANTHROPIC_API_KEY        for provider=anthropic
    OPENAI_API_KEY           for provider=openai (any OpenAI-style endpoint)
    OPENAI_BASE_URL          default https://api.openai.com/v1; for example
                             http://localhost:11434/v1 for a local Ollama
    AGRIPRICE_LLM_MODEL      model name; default claude-haiku-5-5 for
                             anthropic, required for openai

Requests use the standard library (urllib), so no extra package is needed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config as cfg

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TIMEOUT_S = 30

# What the system does not know. The model is told to say so if asked.
NOT_COVERED = [
    "weather, rainfall or harvest forecasts",
    "pests, disease, drying or storage techniques",
    "names or phone numbers of buyers, traders, stores or transporters",
    "actual quotes from service providers (that part of the system is not built yet)",
    "loans, credit or prices of farm inputs",
    "crops or markets not in this payload",
    "prices after the latest recorded month, except the forecasts given",
]


# -----------------------------------------------------------------------------
# Payload
# -----------------------------------------------------------------------------
def _r(x: Any, nd: int = 0) -> Any:
    """Round numbers for the payload; leave anything else alone."""
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return int(round(x)) if nd == 0 else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def build_payload(*, crop: str, market: str, quantity_kg: float, fc, adv, costs,
                  mc=None, alt=None, months_old: int | None = None,
                  is_stale: bool = False) -> dict:
    """Pack the dashboard's results into JSON-safe, rounded figures."""
    opts = adv.options
    options = []
    for _, o in opts.iterrows():
        options.append({
            "option": o["action"],
            "months_stored": int(o["months"]),
            "price_ugx_per_kg": _r(o["price"]),
            "likely_range_ugx_per_kg": None if o["months"] == 0 else [_r(o["low"]), _r(o["high"])],
            "kg_left_to_sell": _r(o["kg_sold"]),
            "storage_cost_ugx": _r(o["storage_ugx"]),
            "expected_earnings_ugx": _r(o["net_ugx"]),
            "difference_vs_selling_now_ugx": _r(o["gain_vs_now_ugx"]),
            "break_even_price_ugx_per_kg": _r(o["breakeven_price"]),
            "chance_beats_selling_now_pct": (None if o["months"] == 0
                                             else _r(o["prob_beats_now"] * 100)),
        })

    forecast_months = []
    for i, d in enumerate(fc.dates):
        forecast_months.append({
            "month": pd.Timestamp(d).strftime("%B %Y"),
            "months_ahead": i + 1,
            "forecast_ugx_per_kg": _r(fc.median[i]),
            "likely_range_ugx_per_kg": [_r(fc.lower[i]), _r(fc.upper[i])],
            "typical_past_error_pct": _r(fc.typical_error_pct[i], 1) if fc.typical_error_pct else None,
        })

    payload: dict = {
        "crop": crop,
        "market": market,
        "quantity_kg": _r(quantity_kg),
        "latest_price": {
            "ugx_per_kg": _r(fc.origin_price),
            "recorded_month": fc.origin_date.strftime("%B %Y"),
            "months_older_than_newest_data": months_old,
            "is_stale": bool(is_stale),
        },
        "recommendation": {
            "action": adv.recommendation,
            "expected_earnings_ugx": _r(adv.expected_net_ugx),
            "earnings_if_sold_now_ugx": _r(adv.net_sell_now_ugx),
            "extra_vs_selling_now_ugx": _r(adv.gain_ugx),
            "confidence_pct": None if adv.confidence is None else _r(adv.confidence * 100),
            "confidence_label": adv.confidence_label,
            "best_storage_option_months": adv.best_store_months,
            "best_storage_option_chance_pct": (None if adv.best_store_prob is None
                                               else _r(adv.best_store_prob * 100)),
            "rule": (f"Storing is advised only if it is expected to earn more AND has at least "
                     f"a {cfg.STORE_MIN_PROBABILITY:.0%} chance of beating selling now."),
        },
        "forecast": {
            "method": fc.label,
            "method_is_same_as_last_month": fc.model == "naive",
            "range_covers_pct_of_past_outcomes": _r(fc.level * 100),
            "months": forecast_months,
            "note": fc.note or None,
        },
        "costs_used": {
            "storage_ugx_per_kg_per_month": _r(costs.storage_ugx_per_kg_month, 2),
            "storage_loss_pct_per_month": _r(costs.loss_rate_per_month * 100, 2),
            "transport_ugx_per_kg_per_km": _r(costs.transport_ugx_per_kg_km, 2),
        },
        "options": options,
        "not_covered_by_this_system": NOT_COVERED,
    }

    if mc is not None:
        t = mc.table
        cmp: dict = {
            "prices_compared_from_month": mc.reference_month.strftime("%B %Y"),
            "markets_compared": int(len(t)),
            "markets_left_out_for_other_months": len(mc.excluded),
            "best_alternative": None,
            "other_markets": [],
        }
        if alt is not None:
            cmp["best_alternative"] = {
                "market": alt["market"],
                "road_distance_km": _r(alt["road_km"]),
                "latest_price_ugx_per_kg": _r(alt["latest_price"]),
                "transport_cost_ugx": _r(alt["transport_ugx"]),
                "best_option_there": alt["best_option"],
                "earnings_after_transport_ugx": _r(alt["best_net_ugx"]),
                "difference_vs_your_recommended_option_ugx": _r(alt["best_net_ugx"] - adv.expected_net_ugx),
            }
        for _, r in t[~t["is_home"]].head(5).iterrows():
            cmp["other_markets"].append({
                "market": r["market"], "road_distance_km": _r(r["road_km"]),
                "latest_price_ugx_per_kg": _r(r["latest_price"]),
                "earnings_after_transport_ugx": _r(r["best_net_ugx"]),
            })
        payload["market_comparison"] = cmp
    return payload


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
_RULES = """You help smallholder farmers in Uganda understand price advice from AgriPrice Advisor.

Rules you must follow:
1. Use ONLY the facts and numbers in the DATA block. Never invent prices, markets, dates,
   costs, buyers, services or chances. Do not do new calculations; quote the figures given.
2. You may round a figure (for example 2,005,500 to about 2,006,000), but write numbers in
   full with commas and the unit (UGX, UGX/kg, kg, km, %). Do not write "million" or "k".
3. Never change or contradict the recommendation in DATA.
4. If the question needs information that is not in DATA, or is listed under
   "not_covered_by_this_system", start your reply with "I don't have that information"
   and say briefly what you can help with instead.
5. Be honest about uncertainty: forecasts can be wrong, and the confidence and chances
   in DATA show how often similar advice was right before.
6. Write in simple, friendly English for a farmer. No headings, no bullet points, no markdown
   tables. Do not give guarantees.
7. If the farmer wants to try different costs or quantities, tell them to change them on the
   crop page instead of calculating new figures yourself.
"""

EXPLAIN_INSTRUCTION = (
    "Write a short explanation (at most 120 words, two or three short paragraphs) of this "
    "advice for the farmer. Start with what to do and the expected earnings. Then explain "
    "why, using the break-even price and the chance that storing pays. If DATA shows a best "
    "alternative market with a positive difference, mention it with the distance and "
    "transport cost. End with one sentence on how reliable the forecast is."
)


def _data_block(payload: dict) -> str:
    return "DATA:\n```json\n" + json.dumps(payload, indent=1, ensure_ascii=False) + "\n```"


# -----------------------------------------------------------------------------
# Grounding check
# -----------------------------------------------------------------------------
# A number in the reply is matched against payload numbers of the SAME unit:
# a price per kg can only match a price per kg, an amount of money only an
# amount, a percentage only a percentage, and so on. Numbers without a unit
# can match anything. This stops a coincidence such as "9,999 UGX/kg"
# passing because a storage cost of 10,000 UGX happens to be close.
_NUM = r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_UNIT = r"\s*(UGX\s*(?:/|per)\s*kg|UGX|shillings|%|percent|per\s*cent|km|kilomet(?:re|er)s?|kg|kilograms?)?"
_NUM_UNIT_RE = re.compile(_NUM + _UNIT, re.IGNORECASE)


def _unit_class(u: str | None) -> str | None:
    if not u:
        return None
    u = re.sub(r"\s+", "", u.lower())
    if u.startswith("ugx/kg") or u.startswith("ugxperkg"):
        return "ugx_per_kg"
    if u in ("ugx", "shillings"):
        return "ugx"
    if u in ("%", "percent") or u.startswith("percent"):
        return "pct"
    if u.startswith("km") or u.startswith("kilomet"):
        return "km"
    if u.startswith("kg") or u.startswith("kilogram"):
        return "kg"
    return None


def _numbers_with_units(text: str) -> list[tuple[float, str | None]]:
    out = []
    for m in _NUM_UNIT_RE.finditer(text):
        whole = m.group(1).replace(",", "")
        val = float(f"{whole}.{m.group(2)}" if m.group(2) else whole)
        out.append((val, _unit_class(m.group(3))))
    return out


def _numbers_in_text(text: str) -> list[float]:
    return [v for v, _ in _numbers_with_units(text)]


def _key_unit(key: str) -> str | None:
    """Unit of a payload value, from its key name (keys are named by unit)."""
    k = key.lower()
    if "ugx_per_kg" in k:
        return "ugx_per_kg"
    if k.endswith("_ugx"):
        return "ugx"
    if "_pct" in k:
        return "pct"
    if k.endswith("_km"):
        return "km"
    if k.endswith("_kg") or k.startswith("kg_"):
        return "kg"
    return None


def _numbers_in_payload(obj: Any, acc: set | None = None, unit: str | None = None) -> set:
    """Set of (value, unit) pairs in the payload; strings are parsed for numbers too."""
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in_payload(v, acc, _key_unit(str(k)))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in_payload(v, acc, unit)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, (int, float)):
        acc.add((float(obj), unit))
    elif isinstance(obj, str):
        acc.update(_numbers_with_units(obj))
    return acc


@dataclass
class GroundingResult:
    ok: bool
    unmatched: list[float] = field(default_factory=list)
    unknown_markets: list[str] = field(default_factory=list)


def _payload_markets(payload: dict) -> set[str]:
    names = {payload.get("market", "")}
    mc = payload.get("market_comparison") or {}
    if mc.get("best_alternative"):
        names.add(mc["best_alternative"]["market"])
    names.update(m["market"] for m in mc.get("other_markets", []))
    return {n for n in names if n}


def check_grounding(text: str, payload: dict, rel_tol: float = 0.01,
                    known_markets: list[str] | None = None) -> GroundingResult:
    """
    Every number in `text` must match a payload number of the same unit within
    1% (rounding is fine). Numbers up to 12 are allowed freely (month counts,
    list positions). If `known_markets` is given, the reply may only name
    markets that appear in the payload.
    """
    pairs = _numbers_in_payload(payload)
    unmatched = []
    for x, unit in _numbers_with_units(text):
        if x <= 12:
            continue
        cands = np.array([v for v, u in pairs if unit is None or u is None or u == unit], dtype=float)
        if cands.size and np.any(np.abs(cands - x) <= np.maximum(rel_tol * np.abs(cands), 1.0)):
            continue
        unmatched.append(x)
    unknown = []
    if known_markets:
        allowed = _payload_markets(payload)
        for m in known_markets:
            if m not in allowed and re.search(rf"\b{re.escape(m)}\b", text):
                unknown.append(m)
    return GroundingResult(ok=not unmatched and not unknown, unmatched=unmatched,
                           unknown_markets=unknown)


# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------
class LLMError(RuntimeError):
    pass


def _post_json(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach the language model: {e}") from None


@dataclass
class AnthropicLLM:
    api_key: str
    model: str = DEFAULT_ANTHROPIC_MODEL
    name: str = "anthropic"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        data = _post_json(ANTHROPIC_URL,
                          {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                           "system": system, "messages": messages},
                          {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION})
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        if not parts:
            raise LLMError("Empty response from the language model.")
        return "".join(parts).strip()


@dataclass
class OpenAICompatibleLLM:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    name: str = "openai"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        data = _post_json(self.base_url.rstrip("/") + "/chat/completions",
                          {"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                           "messages": [{"role": "system", "content": system}, *messages]},
                          {"authorization": f"Bearer {self.api_key}"})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError):
            raise LLMError("Unexpected response from the language model.") from None


def get_llm(settings: Mapping[str, str] | None = None):
    """
    Build a client from settings (Streamlit secrets) and environment variables.
    Returns None if no model is configured.
    """
    def get(key: str, default: str | None = None) -> str | None:
        if settings and settings.get(key):
            return str(settings.get(key))
        return os.environ.get(key, default)

    provider = (get("AGRIPRICE_LLM_PROVIDER") or "").lower()
    if not provider:
        provider = "anthropic" if get("ANTHROPIC_API_KEY") else ("openai" if get("OPENAI_API_KEY") else "none")
    model = get("AGRIPRICE_LLM_MODEL")
    if provider == "anthropic" and get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(api_key=get("ANTHROPIC_API_KEY"), model=model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai" and get("OPENAI_API_KEY") and model:
        return OpenAICompatibleLLM(api_key=get("OPENAI_API_KEY"), model=model,
                                   base_url=get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    return None


# -----------------------------------------------------------------------------
# Explain and answer
# -----------------------------------------------------------------------------
@dataclass
class AssistantReply:
    ok: bool
    text: str
    reason: str = ""                     # why ok is False
    unmatched_numbers: list = field(default_factory=list)   # numbers or market names


def explain(llm, payload: dict, known_markets: list[str] | None = None) -> AssistantReply:
    if llm is None:
        return AssistantReply(False, "", "no language model configured")
    try:
        text = llm.complete(_RULES, [{"role": "user",
                                      "content": _data_block(payload) + "\n\n" + EXPLAIN_INSTRUCTION}],
                            max_tokens=350)
    except LLMError as e:
        return AssistantReply(False, "", str(e))
    g = check_grounding(text, payload, known_markets=known_markets)
    if not g.ok:
        return AssistantReply(False, text, "the reply contained figures or markets not in the "
                              "system's results", g.unmatched + g.unknown_markets)
    return AssistantReply(True, text)


REFUSAL = ("I don't have that information. I can explain this advice, the forecast, the "
           "storage options and the market comparison shown on this page.")


def answer(llm, payload: dict, question: str, history: list[dict] | None = None,
           known_markets: list[str] | None = None) -> AssistantReply:
    """
    Answer a question using only the payload. `history` holds earlier turns as
    [{"role": "user"|"assistant", "content": ...}] for follow-up questions.
    """
    question = (question or "").strip()
    if not question:
        return AssistantReply(False, "", "empty question")
    if llm is None:
        return AssistantReply(False, "", "no language model configured")
    messages = [{"role": "user", "content": _data_block(payload) +
                 "\n\nAnswer the farmer's questions about this advice in at most 100 words."},
                {"role": "assistant", "content": "Understood. I will only use the DATA."}]
    for turn in (history or [])[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question[:500]})
    try:
        text = llm.complete(_RULES, messages, max_tokens=300)
    except LLMError as e:
        return AssistantReply(False, "", str(e))
    g = check_grounding(text, payload, known_markets=known_markets)
    if not g.ok:
        return AssistantReply(False, REFUSAL, "the reply contained figures or markets not in the "
                              "system's results", g.unmatched + g.unknown_markets)
    return AssistantReply(True, text)
