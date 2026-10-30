"""
Forecasters evaluated by the walk-forward backtest.

Every forecaster is a function with the same signature:

    forecast(history: pd.DataFrame, horizon: int) -> np.ndarray

`history` holds one series' rows from the monthly panel up to and including
the forecast origin (columns include date, price, price_observed, segment_id).
The last row is the origin month and always has an observed price. The
function returns `horizon` forecasts for the months after the origin, or NaNs
when it cannot forecast from this history.

Baselines
---------
naive              next months = the latest observed price
seasonal_naive     same month last year, scaled by how the last three months
                   compare with the same three months a year earlier
app_baseline       exactly what the Streamlit app shows today (seasonal-naive
                   plus a straight-line trend)

Models
------
xgb_current        the notebook's XGBoost set-up as it stands before Phase 3
                   (price-level target, `year` feature), run recursively
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Forecaster = Callable[[pd.DataFrame, int], np.ndarray]


def _latest_segment(history: pd.DataFrame) -> pd.Series:
    """Prices since the most recent long gap, indexed by date, no NaNs."""
    seg = history[history["segment_id"] == history["segment_id"].iloc[-1]]
    return seg.set_index("date")["price"].dropna()


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Persistence: every future month equals the latest observed price."""
    last = float(history["price_observed"].iloc[-1])
    return np.full(horizon, last)


def seasonal_naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Same month last year, adjusted for the current price level.

        f(t+h) = p(t+h-12) * mean(p[t-2..t]) / mean(p[t-14..t-12])

    The adjustment carries over inflation and any shift in level since last
    year while keeping the seasonal shape. Falls back to naive when the
    latest segment holds less than 15 months.
    """
    p = _latest_segment(history).to_numpy()
    if len(p) < 15:
        return naive(history, horizon)
    t = len(p) - 1
    recent = p[t - 2:t + 1].mean()
    year_ago = p[t - 14:t - 11].mean()
    ratio = recent / year_ago if year_ago > 0 else 1.0
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        idx = t + h - 12
        # For horizons beyond 12 months this would need two-year lags; not
        # used here (horizons stop at 6) but kept safe.
        base = p[idx] if idx <= t else p[idx - 12]
        out[h - 1] = base * ratio
    return out


def app_baseline(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    The forecast the app displays today (app.py: simple_forecast), reproduced
    so the backtest can say how good the current dashboard numbers are.
    """
    hist = _latest_segment(history)
    if len(hist) < 12:
        last = float(hist.iloc[-1]) if len(hist) else float(history["price_observed"].iloc[-1])
        return np.full(horizon, last)
    last_year = hist.iloc[-12:]
    trend = (hist.iloc[-1] - hist.iloc[-13]) / 12 if len(hist) >= 13 else 0.0
    return np.array([max(0.0, float(last_year.iloc[i % 12]) + trend * (i + 1))
                     for i in range(horizon)])


# -----------------------------------------------------------------------------
# Current XGBoost (as in the notebook before Phase 3)
# -----------------------------------------------------------------------------
XGB_FEATURES = [
    "price_lag1", "price_lag2", "price_lag3", "price_lag6", "price_lag12",
    "roll_mean_3", "roll_mean_6", "roll_std_3",
    "month_sin", "month_cos", "year",
]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)


def xgb_features(price: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    The notebook's feature set. On the complete monthly calendar, shift(k)
    is exactly k months back, and NaNs inside long gaps make the affected
    rows drop out instead of silently using the wrong month.
    """
    p = pd.Series(price.to_numpy(), dtype=float)
    f = pd.DataFrame({
        "price_lag1": p.shift(1), "price_lag2": p.shift(2),
        "price_lag3": p.shift(3), "price_lag6": p.shift(6),
        "price_lag12": p.shift(12),
        "roll_mean_3": p.shift(1).rolling(3).mean(),
        "roll_mean_6": p.shift(1).rolling(6).mean(),
        "roll_std_3": p.shift(1).rolling(3).std(),
    })
    d = pd.to_datetime(pd.Series(dates.to_numpy()))
    month = d.dt.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    f["year"] = d.dt.year
    return f


class XGBCurrent:
    """
    Recursive multi-step XGBoost with the notebook's set-up.

    Refitting at every origin is the honest choice but slow, so the model is
    refitted every `refit_every` origins and reused in between. Between
    refits the model only sees older training data, which can only make it
    look worse, never better, so the comparison stays fair to the baselines.
    """

    name = "xgb_current"

    def __init__(self, refit_every: int = 3, min_train_rows: int = 24):
        self.refit_every = refit_every
        self.min_train_rows = min_train_rows
        self._cache: dict = {}

    def __call__(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        import xgboost as xgb

        key = (history["commodity"].iloc[0], history["market"].iloc[0])
        n = len(history)
        cached = self._cache.get(key)
        if cached is None or n - cached[0] >= self.refit_every or n < cached[0]:
            feats = xgb_features(history["price"], history["date"])
            train = feats.assign(y=history["price"].to_numpy()).dropna()
            if len(train) < self.min_train_rows:
                return np.full(horizon, np.nan)
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(train[XGB_FEATURES], train["y"])
            self._cache[key] = (n, model)
        model = self._cache[key][1]

        # Recursive forecasting: feed each prediction back in as a lag.
        prices = list(history["price"].to_numpy(dtype=float))
        dates = list(pd.to_datetime(history["date"]))
        out = []
        for _ in range(horizon):
            dates.append(dates[-1] + pd.offsets.MonthBegin(1))
            prices.append(np.nan)
            feats = xgb_features(pd.Series(prices), pd.Series(dates)).iloc[[-1]]
            if feats[XGB_FEATURES].isna().any(axis=1).iloc[0]:
                return np.full(horizon, np.nan)
            yhat = float(model.predict(feats[XGB_FEATURES])[0])
            prices[-1] = yhat
            out.append(yhat)
        return np.array(out)


MODEL_LABELS = {
    "naive": "Same as last month (naive)",
    "seasonal_naive": "Same month last year, adjusted (seasonal naive)",
    "app_baseline": "Old dashboard forecast",
    "xgb_current": "XGBoost",
}


def get_forecaster(name: str) -> Forecaster:
    """Look up a forecaster by the name used in the backtest and registry."""
    if name == "xgb_current":
        return XGBCurrent(refit_every=1)
    models = default_forecasters(include_xgb=False)
    if name not in models:
        raise KeyError(f"Unknown forecaster '{name}'. Add it to src/forecasters.py.")
    return models[name]


def default_forecasters(include_xgb: bool = True) -> dict[str, Forecaster]:
    models: dict[str, Forecaster] = {
        "naive": naive,
        "seasonal_naive": seasonal_naive,
        "app_baseline": app_baseline,
    }
    if include_xgb:
        models["xgb_current"] = XGBCurrent()
    return models
