"""
Tests for the user-flow logic (Phase 6).

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import flow  # noqa: E402


@pytest.mark.parametrize("raw", ["0772 123 456", "0772123456", "772123456", "+256772123456",
                                 "256 772 123 456", "00256-772-123-456", " (0772) 123-456 "])
def test_valid_phone_forms(raw):
    assert flow.normalise_ug_phone(raw) == "+256772123456"


@pytest.mark.parametrize("raw", ["", "12345", "0412123456", "+254712345678", "07721234567",
                                 "077212345", "phone", "+2567721234"])
def test_invalid_phones(raw):
    assert flow.normalise_ug_phone(raw) is None


@pytest.fixture
def summary_and_markets():
    summary = pd.DataFrame({
        "commodity": ["Maize", "Maize", "Maize", "Beans", "Millet"],
        "market":    ["Gulu", "Lira", "Owino", "Lira", "Gulu"],
        "pricetype": ["Retail"] * 5,
        "is_stale":  [False, False, True, False, True],
    })
    markets = pd.DataFrame({
        "market": ["Gulu", "Lira", "Owino"],
        "region": ["Northern", "Northern", "Central"],
        "latitude": [2.78, 2.23, 0.32], "longitude": [32.30, 32.91, 32.57],
    })
    return summary, markets


def test_only_crops_and_markets_with_recent_data(summary_and_markets):
    summary, _ = summary_and_markets
    assert flow.available_crops(summary) == ["Maize", "Beans"]     # millet only stale
    assert flow.available_markets(summary, "Maize") == ["Gulu", "Lira"]   # Owino stale


def test_nearest_market_and_coverage(summary_and_markets):
    _, markets = summary_and_markets
    near_gulu = flow.nearest_market(2.75, 32.33, markets, ["Gulu", "Lira"])
    assert near_gulu.market == "Gulu" and near_gulu.in_coverage and near_gulu.distance_km < 10
    moscow = flow.nearest_market(55.75, 37.62, markets, ["Gulu", "Lira"])
    assert not moscow.in_coverage


def test_resolve_market_priority(summary_and_markets):
    summary, markets = summary_and_markets
    # About 87 km from Owino and 130 km from Lira. Owino is nearest but stale for
    # maize, so the nearest market with recent maize prices (Lira) is used.
    between = (1.1, 32.6)
    auto = flow.resolve_market("Maize", summary, markets, between, manual=None)
    assert auto.source == "location" and auto.market == "Lira"
    # a valid manual choice wins
    assert flow.resolve_market("Maize", summary, markets, between, "Gulu").market == "Gulu"
    # an invalid manual choice falls back to location, with a note
    r = flow.resolve_market("Beans", summary, markets, (2.7, 32.3), "Gulu")
    assert r.market == "Lira" and r.source == "location" and "no recent Beans" in r.note
    # no location and no choice: ask the farmer
    assert flow.resolve_market("Maize", summary, markets, None, None).source == "none"
    # outside coverage: ask the farmer, with a note
    far = flow.resolve_market("Maize", summary, markets, (55.75, 37.62), None)
    assert far.market is None and "km from the nearest market" in far.note


@pytest.mark.parametrize("raw,expected", [
    ("Name@Example.com", "name@example.com"),
    ("  farmer.one+prices@mail.co.ug ", "farmer.one+prices@mail.co.ug"),
])
def test_valid_emails(raw, expected):
    assert flow.normalise_email(raw) == expected
    assert flow.parse_contact(raw) == ("email", expected)


@pytest.mark.parametrize("raw", ["name@", "@example.com", "name@example", "name@@example.com",
                                 "na me@example.com", "name@example..com", "name..x@example.com"])
def test_invalid_emails(raw):
    assert flow.normalise_email(raw) is None
    assert flow.parse_contact(raw) is None


def test_parse_contact_phone_and_garbage():
    assert flow.parse_contact("0772 123 456") == ("phone", "+256772123456")
    assert flow.parse_contact("hello") is None
    assert flow.parse_contact("") is None
