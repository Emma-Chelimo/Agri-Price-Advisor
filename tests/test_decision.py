"""
Tests for the sell-or-store maths and market comparison (Phase 5).

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import data, decision as D, forecasting as fz  # noqa: E402
from tests.test_evaluation import make_cleaned, monthly_prices  # noqa: E402


def make_fc(prices, sigma=0.1, p0=1000.0):
    w = sigma * fz.band_z(0.8)
    n = len(prices)
    return fz.Forecast(
        crop="Maize", market="Home", model="test", label="test",
        origin_date=pd.Timestamp("2026-01-01"), origin_price=p0,
        dates=list(pd.date_range("2026-02-01", periods=n, freq="MS")),
        median=list(prices), lower=[p * np.exp(-w) for p in prices],
        upper=[p * np.exp(w) for p in prices], level=0.8,
        log_sigma=[sigma] * n, typical_error_pct=[10.0] * n,
    )


def test_haversine_owino_to_gulu():
    # Owino (0.32, 32.57) to Gulu (2.78, 32.30): about 275 km in a straight line
    assert D.haversine_km(0.32, 32.57, 2.78, 32.30) == pytest.approx(275, abs=5)


def test_break_even_and_net_formulas():
    costs = D.Costs(storage_ugx_per_kg_month=20, loss_rate_per_month=0.01)
    o = D.timing_options(make_fc([1100, 1200]), qty=100, costs=costs).set_index("months")
    assert o.at[0, "net_ugx"] == 100_000
    surviving = 100 * 0.99
    assert o.at[1, "net_ugx"] == pytest.approx(1100 * surviving - 20 * 100)
    assert o.at[1, "breakeven_price"] == pytest.approx((100_000 + 2000) / surviving)


def test_flat_forecast_zero_costs_is_a_coin_flip_so_sell_now():
    costs = D.Costs(0, 0)
    adv = D.advise(make_fc([1000] * 6), qty=500, costs=costs)
    assert adv.recommendation == "Sell now"
    assert adv.best_store_prob == pytest.approx(0.5)
    assert adv.confidence == pytest.approx(0.5)


def test_clear_rise_recommends_storing_with_high_confidence():
    costs = D.Costs(storage_ugx_per_kg_month=10, loss_rate_per_month=0.005)
    adv = D.advise(make_fc([1050, 1150, 1300, 1300, 1250, 1200], sigma=0.05), 500, costs)
    assert adv.recommendation == "Store 3 months"
    assert adv.gain_ugx > 0 and adv.confidence > cfg.STORE_MIN_PROBABILITY


def test_small_gain_with_wide_uncertainty_still_sells_now():
    costs = D.Costs(storage_ugx_per_kg_month=5, loss_rate_per_month=0.0)
    adv = D.advise(make_fc([1030] * 6, sigma=0.4), 500, costs)
    top = adv.options.loc[adv.options["net_ugx"].idxmax()]
    assert top["months"] > 0                      # median says store...
    assert adv.recommendation == "Sell now"       # ...but the chance is too low
    assert adv.confidence < 0.6


def test_no_band_means_no_confidence():
    fc = make_fc([1000] * 3)
    fc.log_sigma = [np.nan] * 3
    adv = D.advise(fc, 100, D.Costs(20, 0.01))
    assert adv.confidence is None and adv.confidence_label == "Not available"


@pytest.fixture
def three_markets():
    rows = []
    for name, lat, lon, level, end in [("Home", 0.0, 32.0, 1000, "2025-12"),
                                       ("Near", 0.0, 32.5, 1300, "2025-12"),
                                       ("Far", 3.0, 32.0, 1600, "2025-12"),
                                       ("Old", 0.1, 32.1, 5000, "2025-11")]:
        vals = monthly_prices("2020-01", [level] * 72)
        vals = {k: v for k, v in vals.items() if k <= end}
        c = make_cleaned(vals, market=name)
        c["latitude"], c["longitude"] = lat, lon
        rows.append(c)
    cleaned = pd.concat(rows, ignore_index=True)
    panel = data.build_monthly_panel(cleaned)
    markets = cleaned.groupby("market", as_index=False).agg(
        region=("region", "first"), latitude=("latitude", "first"), longitude=("longitude", "first"))
    return panel, markets


def test_market_comparison_counts_transport_and_same_month_rule(three_markets):
    panel, markets = three_markets
    costs = D.Costs(20, 0.01, transport_ugx_per_kg_km=3.0)
    mc = D.compare_markets(panel, markets, "Maize", "Home", 100, 3, costs, registry={"crops": {}})
    t = mc.table.set_index("market")
    assert "Old" in mc.excluded and "Old" not in t.index      # different month
    assert t.at["Home", "transport_ugx"] == 0
    near_km = D.haversine_km(0, 32, 0, 32.5) * cfg.ROAD_FACTOR
    assert t.at["Near", "transport_ugx"] == pytest.approx(3.0 * round(near_km) * 100)
    assert t.at["Near", "best_net_ugx"] == pytest.approx(1300 * 100 - t.at["Near", "transport_ugx"])
    # Far pays more per kg but loses after ~430 km of transport
    assert t.at["Far", "best_net_ugx"] < t.at["Near", "best_net_ugx"]
    assert D.best_alternative(mc)["market"] == "Near"


def test_sudden_jump_rule():
    assert D.sudden_jump(2550, 5917)          # Gulu millet, Jan 2024
    assert not D.sudden_jump(2550, 2750)
    assert not D.sudden_jump(None, 5000)


def test_market_with_sudden_jump_is_not_suggested(three_markets):
    panel, markets = three_markets
    # make Near's latest price a one-month spike
    i = panel.index[(panel["market"] == "Near") & (panel["date"] == panel["date"].max())]
    panel.loc[i, ["price", "price_observed"]] = 3000.0
    costs = D.Costs(20, 0.01, transport_ugx_per_kg_km=3.0)
    mc = D.compare_markets(panel, markets, "Maize", "Home", 100, 3, costs, registry={"crops": {}})
    t = mc.table.set_index("market")
    assert t.at["Near", "sudden_jump"] and not t.at["Far", "sudden_jump"]
    assert D.best_alternative(mc)["market"] == "Far"
