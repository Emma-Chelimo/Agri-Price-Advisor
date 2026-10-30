"""
Tests for model selection and forecast serving (Phase 4).

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import data, forecasting as fz, forecasters as F  # noqa: E402
from tests.test_evaluation import make_cleaned, monthly_prices  # noqa: E402


def fake_backtest(skill_by_model: dict[str, float], crop="Maize", n=60) -> pd.DataFrame:
    """Validation forecasts where each model's error is scaled to give a set skill."""
    rng = np.random.default_rng(0)
    rows = []
    for h in cfg.HORIZONS:
        actual = 1000 + rng.normal(0, 50, n)
        naive_err = rng.normal(0, 100, n)
        for model, skill in skill_by_model.items():
            err = naive_err * (1 - skill)
            for i in range(n):
                rows.append(("validation", model, crop, f"M{i % 5}",
                             pd.Timestamp("2022-01-01") + pd.DateOffset(months=i % 20),
                             h, None, 1000.0, actual[i], actual[i] + err[i]))
    cols = ["split", "model", "commodity", "market", "origin", "horizon",
            "target_date", "origin_price", "actual", "forecast"]
    return pd.DataFrame(rows, columns=cols)


def test_selection_keeps_naive_unless_margin_beaten():
    reg = fz.build_registry(fake_backtest({"naive": 0, "xgb_current": 0.01}), write=False)
    assert reg["crops"]["Maize"]["model"] == "naive"
    reg = fz.build_registry(fake_backtest({"naive": 0, "xgb_current": 0.30}), write=False)
    assert reg["crops"]["Maize"]["model"] == "xgb_current"


def test_app_baseline_never_selected():
    reg = fz.build_registry(fake_backtest({"naive": 0, "app_baseline": 0.5}), write=False)
    assert reg["crops"]["Maize"]["model"] == "naive"


def test_crop_without_results_falls_back_to_naive():
    reg = fz.build_registry(fake_backtest({"naive": 0, "xgb_current": 0.3}), write=False)
    assert reg["crops"]["Beans"]["model"] == "naive"


def _panel(last_month_missing=False):
    vals = monthly_prices("2016-01", np.linspace(1000, 2000, 120))
    if last_month_missing:
        vals.pop(list(vals)[-2])          # gap just before the end
    return data.build_monthly_panel(make_cleaned(vals))


def test_forecast_starts_after_latest_observed_price_and_clamps_horizon():
    reg = fz.build_registry(fake_backtest({"naive": 0}), write=False)
    p = _panel()
    fc = fz.forecast(p, "Maize", "Testville", horizon=12, registry=reg)
    assert len(fc.dates) == cfg.MAX_FORECAST_HORIZON
    assert fc.origin_date == p["date"].max()
    assert fc.dates[0] == fc.origin_date + pd.offsets.MonthBegin(1)
    assert np.allclose(fc.median, fc.origin_price)


def test_bands_contain_forecast():
    reg = fz.build_registry(fake_backtest({"naive": 0}), write=False)
    fc = fz.forecast(_panel(), "Maize", "Testville", horizon=6, registry=reg)
    lo, mid, hi = map(np.array, (fc.lower, fc.median, fc.upper))
    assert np.all(lo <= mid) and np.all(mid <= hi)


def test_missing_registry_uses_naive_without_bands():
    fc = fz.forecast(_panel(), "Maize", "Testville", horizon=3, registry={"crops": {}})
    assert fc.model == "naive" and fc.note
    assert np.all(np.isnan(fc.lower))


def test_failed_model_falls_back_to_naive(monkeypatch):
    reg = fz.build_registry(fake_backtest({"naive": 0, "xgb_current": 0.3}), write=False)
    monkeypatch.setattr(F, "get_forecaster",
                        lambda name: (lambda h, n: np.full(n, np.nan)) if name != "naive" else F.naive)
    fc = fz.forecast(_panel(), "Maize", "Testville", horizon=3, registry=reg)
    assert fc.model == "naive" and fc.used_fallback
