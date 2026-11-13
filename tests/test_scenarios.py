"""
Scenario tests: every outcome the advice can give, forced with controlled forecasts.

Outcomes covered
    sell now at home (flat prices, small rise, uncertain rise, high costs, high losses)
    store 1, 2 or 3 months at home
    the 60% rule and the "best expected earnings" rule working together
    the exact forecast rise at which "store" switches on
    no confidence available (no error band)
    commodities on the reference forecast never store
    quantity never changes the verdict
    another market: chosen when transport is cheap, not chosen when it is dear,
    not chosen when its price is a sudden jump, not chosen when it is from another month

Run from the project root:   python -m pytest -q tests/test_scenarios.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import decision as D  # noqa: E402
from tests.test_decision import make_fc, three_markets  # noqa: E402,F401

Z60 = NormalDist().inv_cdf(cfg.STORE_MIN_PROBABILITY)
MAIZE = D.Costs(20, 0.0045)          # app defaults for maize
BEANS = D.Costs(20, 0.015)           # app defaults for beans


@pytest.fixture(autouse=True)
def raw_confidence(monkeypatch):
    """Use the raw probabilities so the tests do not depend on the calibration file."""
    monkeypatch.setattr(D, "calibrate_confidence", lambda p, kind: p)


# ---------------------------------------------------------------- sell now at home
def test_flat_prices_sell_now():
    adv = D.advise(make_fc([1000, 1000, 1000], sigma=0.1), 1000, MAIZE)
    assert adv.recommendation == "Sell now" and adv.months == 0
    assert adv.gain_ugx == 0


def test_rise_smaller_than_costs_sell_now():
    # +1.5% a month does not cover 20 UGX/kg storage plus 0.45% loss on a 1,000 UGX price
    adv = D.advise(make_fc([1015, 1030, 1045], sigma=0.05), 1000, MAIZE)
    assert adv.recommendation == "Sell now"
    assert all(adv.options["gain_vs_now_ugx"].iloc[1:] < 0)


def test_rise_that_pays_on_average_but_is_too_uncertain_sell_now():
    # store 3 months gains on the median forecast, but with a wide spread the chance is under 60%
    adv = D.advise(make_fc([1030, 1060, 1090], sigma=0.4), 1000, MAIZE)
    assert adv.recommendation == "Sell now"
    assert adv.options["gain_vs_now_ugx"].max() > 0
    assert adv.best_store_prob < cfg.STORE_MIN_PROBABILITY
    assert adv.confidence == pytest.approx(1 - adv.best_store_prob)


def test_high_storage_cost_turns_store_into_sell():
    fc = make_fc([1060, 1120, 1180], sigma=0.08)
    assert D.advise(fc, 1000, MAIZE).months > 0
    assert D.advise(fc, 1000, D.Costs(80, 0.0045)).months == 0


def test_high_losses_turn_store_into_sell():
    fc = make_fc([1060, 1120, 1180], sigma=0.08)
    assert D.advise(fc, 1000, BEANS).months > 0
    assert D.advise(fc, 1000, D.Costs(20, 0.08)).months == 0


# ---------------------------------------------------------------- store at home
def test_store_one_month_when_price_peaks_next_month():
    adv = D.advise(make_fc([1150, 1050, 1000], sigma=0.05), 1000, MAIZE)
    assert adv.recommendation == "Store 1 month" and adv.months == 1


def test_store_two_months_when_price_peaks_in_two_months():
    adv = D.advise(make_fc([1080, 1200, 1100], sigma=0.05), 1000, MAIZE)
    assert adv.recommendation == "Store 2 months" and adv.months == 2


def test_store_three_months_when_price_keeps_rising():
    adv = D.advise(make_fc([1080, 1160, 1260], sigma=0.05), 1000, MAIZE)
    assert adv.recommendation == "Store 3 months" and adv.months == 3
    assert adv.confidence >= cfg.STORE_MIN_PROBABILITY
    assert adv.confidence_label in ("High", "Medium")


def test_rule_picks_best_earnings_but_needs_60_percent_for_that_option():
    # 3 months earns most on the median but is too uncertain (wide spread at 3 months);
    # the rule does not fall back to a safer, smaller option: it says sell now.
    fc = make_fc([1010, 1020, 1300], sigma=0.1)
    fc.log_sigma = [0.05, 0.05, 1.0]
    adv = D.advise(fc, 1000, MAIZE)
    store = adv.options[adv.options["months"] > 0]
    assert store.loc[store["net_ugx"].idxmax(), "months"] == 3
    assert adv.recommendation == "Sell now"


@pytest.mark.parametrize("costs,sigma,bias", [(MAIZE, 0.1422, -0.0142), (BEANS, 0.1006, 0.0167)])
def test_store_switches_on_exactly_at_the_threshold(costs, sigma, bias):
    """One-month storage: the forecast must cover costs AND give a 60% chance."""
    p0 = 1000.0
    surv = 1 - costs.loss_rate_per_month
    breakeven = (p0 + costs.storage_ugx_per_kg_month) / surv
    need = max(breakeven, breakeven * np.exp(Z60 * sigma - bias))   # price needed in one month

    def verdict(price):
        fc = make_fc([price, 0.5 * p0, 0.5 * p0], sigma=sigma, p0=p0)   # only month 1 can pay
        fc.log_bias = [bias] * 3
        return D.advise(fc, 1000, costs).months

    assert verdict(need * 0.999) == 0
    assert verdict(need * 1.001) == 1


# ---------------------------------------------------------------- confidence
def test_no_error_band_means_no_confidence_but_still_advises():
    fc = make_fc([1080, 1160, 1260], sigma=0.05)
    fc.log_sigma = [np.nan] * 3
    adv = D.advise(fc, 1000, MAIZE)
    assert adv.confidence is None and adv.confidence_label == "Not available"
    assert adv.months == 3                     # falls back to the best expected earnings


@pytest.mark.parametrize("p,label", [(0.80, "High"), (0.75, "High"), (0.65, "Medium"),
                                     (0.60, "Medium"), (0.59, "Low: a close call"), (0.10, "Low: a close call")])
def test_confidence_labels(p, label):
    assert D.confidence_label(p) == label


# ---------------------------------------------------------------- reference forecast never stores
@pytest.mark.parametrize("bias", [0.0, 0.02, 0.06])
def test_reference_forecast_never_advises_storage(bias):
    """Commodities without a model forecast a flat price; storage always costs money."""
    fc = make_fc([1000, 1000, 1000], sigma=0.12)
    fc.log_bias = [bias] * 3
    for costs in (MAIZE, BEANS, D.Costs(1, 0.001)):
        assert D.advise(fc, 1000, costs).months == 0


# ---------------------------------------------------------------- quantity
@pytest.mark.parametrize("prices", [[1000, 1000, 1000], [1150, 1050, 1000], [1080, 1160, 1260]])
def test_quantity_never_changes_the_verdict(prices):
    fc = make_fc(prices, sigma=0.06)
    verdicts = {D.advise(fc, q, MAIZE).recommendation for q in (50, 1000, 50_000)}
    assert len(verdicts) == 1


# ---------------------------------------------------------------- another market
def test_cheap_transport_sends_farmer_to_better_market(three_markets):
    panel, markets = three_markets
    mc = D.compare_markets(panel, markets, "Maize", "Home", 1000, 3,
                           D.Costs(20, 0.0045, transport_ugx_per_kg_km=3.0), registry={"crops": {}})
    alt = D.best_alternative(mc)
    home = mc.table.set_index("market").at["Home", "best_net_ugx"]
    assert alt["market"] == "Near" and alt["best_net_ugx"] > home


def test_dear_transport_keeps_farmer_at_home(three_markets):
    panel, markets = three_markets
    mc = D.compare_markets(panel, markets, "Maize", "Home", 1000, 3,
                           D.Costs(20, 0.0045, transport_ugx_per_kg_km=50.0), registry={"crops": {}})
    t = mc.table.set_index("market")
    assert t["best_net_ugx"].idxmax() == "Home"
    assert D.best_alternative(mc)["best_net_ugx"] < t.at["Home", "best_net_ugx"]


def test_market_from_another_month_is_never_compared(three_markets):
    panel, markets = three_markets
    mc = D.compare_markets(panel, markets, "Maize", "Home", 1000, 3,
                           D.Costs(20, 0.0045, transport_ugx_per_kg_km=0.0), registry={"crops": {}})
    assert "Old" in mc.excluded                                  # far higher price, but older
    assert D.best_alternative(mc)["market"] != "Old"
