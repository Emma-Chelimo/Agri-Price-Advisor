"""
Sanity tests for the data pipeline and the walk-forward backtest.

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
from src import data, evaluation as ev, forecasters as F  # noqa: E402


def make_cleaned(prices: dict[str, float], commodity="Maize", market="Testville"):
    """Build a fake cleaned table from {'YYYY-MM': price}."""
    periods = pd.PeriodIndex(list(prices), freq="M")
    return pd.DataFrame({
        "commodity": commodity, "market": market, "pricetype": "Retail",
        "market_id": 1, "admin1": "Lira", "region": "Northern",
        "latitude": 2.2, "longitude": 32.9,
        "period": periods, "date": periods.to_timestamp(),
        "price": list(prices.values()),
    })


def monthly_prices(start: str, values) -> dict[str, float]:
    p = pd.period_range(start, periods=len(values), freq="M")
    return {str(k): float(v) for k, v in zip(p, values)}


@pytest.fixture
def seasonal_panel():
    # Pure multiplicative growth times a fixed seasonal pattern, 2015-2026.
    n = 136
    t = np.arange(n)
    season = 1 + 0.15 * np.sin(2 * np.pi * t / 12)
    values = 1000 * (1.01 ** t) * season
    return data.build_monthly_panel(make_cleaned(monthly_prices("2015-01", values)))


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def test_short_gaps_filled_long_gaps_split():
    vals = monthly_prices("2018-01", np.linspace(1000, 2000, 48))
    for m in ["2019-03", "2019-04"]:          # 2-month gap: interpolated
        vals.pop(m)
    for m in ["2020-01", "2020-02", "2020-03", "2020-04"]:   # 4 months: long
        vals.pop(m)
    p = data.build_monthly_panel(make_cleaned(vals))
    assert p["date"].diff().dropna().dt.days.between(28, 31).all()
    assert p.loc[p["is_imputed"], "date"].dt.strftime("%Y-%m").tolist() == ["2019-03", "2019-04"]
    assert p["in_long_gap"].sum() == 4 and p.loc[p["in_long_gap"], "price"].isna().all()
    assert p["segment_id"].max() == 1


# -----------------------------------------------------------------------------
# No look-ahead
# -----------------------------------------------------------------------------
def test_forecaster_never_sees_future(seasonal_panel):
    seen = []

    def spy(history, horizon):
        seen.append((history["date"].iloc[-1], history["date"].max()))
        return np.full(horizon, 1.0)

    fc = ev.backtest(seasonal_panel, {"naive": F.naive, "spy": spy},
                     split="test", verbose=False)
    assert seen and all(last == latest for last, latest in seen)
    # every forecast's target is after its origin
    assert (fc["target_date"] > fc["origin"]).all()


def test_split_boundaries(seasonal_panel):
    val = ev.backtest(seasonal_panel, {"naive": F.naive}, split="validation", verbose=False)
    test = ev.backtest(seasonal_panel, {"naive": F.naive}, split="test", verbose=False)
    t0 = pd.Period(cfg.TEST_START, "M").to_timestamp()
    assert (val["target_date"] < t0).all() and (val["origin"] < t0).all()
    assert (test["origin"] >= t0).all()


def test_interpolated_months_never_targets():
    vals = monthly_prices("2016-01", np.linspace(1000, 3000, 120))
    vals.pop("2024-06")
    p = data.build_monthly_panel(make_cleaned(vals))
    fc = ev.backtest(p, {"naive": F.naive}, split="test", verbose=False)
    assert pd.Timestamp("2024-06-01") not in set(fc["target_date"])
    assert pd.Timestamp("2024-06-01") not in set(fc["origin"])


# -----------------------------------------------------------------------------
# Baselines and metrics
# -----------------------------------------------------------------------------
def test_naive_perfect_on_flat_series():
    p = data.build_monthly_panel(make_cleaned(monthly_prices("2016-01", [1500] * 120)))
    fc = ev.backtest(p, {"naive": F.naive}, split="test", verbose=False)
    assert np.allclose(fc["forecast"], fc["actual"])


def test_seasonal_naive_exact_on_growth_times_season(seasonal_panel):
    fc = ev.backtest(seasonal_panel, {"naive": F.naive, "seasonal_naive": F.seasonal_naive},
                     split="test", verbose=False)
    sn = fc[fc["model"] == "seasonal_naive"]
    assert np.allclose(sn["forecast"], sn["actual"], rtol=1e-9)
    s = ev.summary(fc)
    assert (s.loc[s["model"] == "seasonal_naive", "pooled_skill"] > 0.99).all()
    assert (s.loc[s["model"] == "naive", "pooled_skill"] == 0).all()


def test_common_pairs_drops_failed_forecasts(seasonal_panel):
    def sometimes_nan(history, horizon):
        out = np.full(horizon, 1.0)
        if history["date"].iloc[-1].month == 3:
            out[:] = np.nan
        return out

    fc = ev.backtest(seasonal_panel, {"naive": F.naive, "flaky": sometimes_nan},
                     split="test", verbose=False)
    assert not (fc["origin"].dt.month == 3).any()
    assert fc.groupby("model").size().nunique() == 1
