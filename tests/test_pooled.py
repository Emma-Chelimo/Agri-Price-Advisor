"""
Tests for the pooled model (src/pooled.py) and the decision metrics.

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
from src import data, evaluation as ev, pooled as P  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return data.load_monthly()


@pytest.fixture(scope="module")
def feats(panel):
    return P.build_features(panel)


def test_features_never_use_later_prices(panel, feats):
    """Cutting the data off at a date must not change any feature up to that date."""
    cut = pd.Timestamp("2021-03-01")
    part = P.build_features(panel[panel["date"] <= cut])
    full = feats[feats.index.get_level_values("date") <= cut][P.FEATURES]
    pd.testing.assert_frame_equal(part[P.FEATURES].reindex(full.index), full, check_dtype=False)


def test_targets_are_real_prices_only(feats):
    """A target exists only where both months have a recorded (not filled-in) price."""
    f = feats.reset_index()
    has_y = f["y3"].notna()
    assert (f.loc[has_y, "age"] == 0).all()
    later = f.groupby(["commodity", "market"])["lp_obs"].shift(-3)
    assert later[has_y].notna().all()


def test_model_trains_only_on_targets_before_its_fit_date(panel, monkeypatch):
    seen = {}
    real_fit = P._Ensemble.fit

    def spy(self, train, y):
        seen["latest_target"] = train["tdate3"].max()
        return real_fit(self, train, y)

    monkeypatch.setattr(P._Ensemble, "fit", spy)
    m = P.PooledGBM(panel, refit_months=3)
    s = panel[(panel["commodity"] == "Beans") & (panel["market"] == "Gulu")
              & (panel["date"] <= "2022-05-01")]
    s = s.loc[: s.index[s["price_observed"].notna()][-1]]
    out = m(s, 3)
    assert np.isfinite(out).all()
    assert seen["latest_target"] <= pd.Timestamp("2022-04-01")   # quarter start before May
    assert P.fit_date_for(pd.Timestamp("2022-05-01"), 3) == pd.Timestamp("2022-04-01")


def test_shrinkage_limits():
    base = type("B", (), {"predict_log_change": lambda self, c, m, o, h: np.full(h, 0.2)})()
    hist = pd.DataFrame({"commodity": ["Maize"], "market": ["Gulu"],
                         "date": [pd.Timestamp("2024-01-01")], "price_observed": [1000.0]})
    naive_like = P.PooledShrunk(base, {"Maize": {str(h): 0.0 for h in range(1, 7)}})
    raw_like = P.PooledShrunk(base, {"Maize": {str(h): 1.0 for h in range(1, 7)}})
    assert np.allclose(naive_like(hist, 6), 1000.0)
    assert np.allclose(raw_like(hist, 6), 1000.0 * np.exp(0.2))


def test_fit_lambdas_picks_zero_for_a_useless_model():
    rng = np.random.default_rng(0)
    n = 400
    p0 = np.full(n, 1000.0)
    actual = p0 * np.exp(rng.normal(0, 0.1, n))
    noise_fc = p0 * np.exp(rng.normal(0, 0.3, n))           # pure noise
    fc = pd.DataFrame({"model": "pooled_gbm", "commodity": "Maize", "horizon": 1,
                       "origin_price": p0, "actual": actual, "forecast": noise_fc})
    assert P.fit_lambdas(fc)["Maize"]["1"] <= 0.1
    good = pd.DataFrame({"model": "pooled_gbm", "commodity": "Maize", "horizon": 1,
                         "origin_price": p0, "actual": actual, "forecast": actual})
    assert P.fit_lambdas(good)["Maize"]["1"] == 1.0


def test_decision_metrics():
    rows = []
    for split in ("validation", "test"):
        for i in range(10):
            up = i % 2 == 0
            rows.append({"split": split, "model": "m", "commodity": "Maize", "horizon": 1,
                         "origin_price": 100.0, "forecast": 110.0 if up else 101.0,
                         "actual": 112.0 if i < 8 else 90.0})
    dm = ev.decision_metrics(pd.DataFrame(rows))
    t = dm[dm["split"] == "test"].iloc[0]
    assert t["confident_share"] == 0.5                      # 5 of 10 predict a 10% rise
    assert t["confident_dir_acc"] == 0.8                     # 4 of those 5 rose
    assert t["up_call_hit"] == 0.8
    assert 0.0 <= t["band_coverage"] <= 1.0
    assert cfg.CONFIDENT_MOVE == 0.05
