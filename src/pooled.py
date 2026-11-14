"""
Pooled, direct, change-based forecaster (Phase 3).

What is different from the notebook's XGBoost (forecasters.XGBCurrent)
---------------------------------------------------------------------
1. Pooled. One model per crop and horizon, trained on every market at once,
   with the region as an input. A typical market has about 50 real monthly
   prices; a crop pooled over 25 markets has over a thousand.
2. Change, not level. The target is log(price in h months / price now). Trees
   cannot predict outside the range they were trained on, so a price-level
   target pulls forecasts back to old prices after inflation; a change target
   does not.
3. Direct. A separate model for each horizon 1..6, so errors are not fed back
   in month by month.
4. Cross-market features. The market's gap to the national and regional
   average for the crop, its gap to its own 12-month average, volatility, the
   national 12-month change, the month, and the crop's usual seasonal change
   from this month (learnt from the training rows only).
5. Two models averaged: a small tree model (XGBoost, depth 2) and a robust
   linear model (Huber regression). Their errors partly cancel.

Short-term momentum features (last 1-3 month changes) were tried on the
validation split and dropped from the tree model: they made it extrapolate
turning points and lose to naive at 4-6 months.

No look-ahead
-------------
* Features use only real (observed) prices. A missing month carries the last
  real price forward for at most two months; it is never interpolated from a
  later price. National and regional averages use the same carried values on
  the same date, so a feature for month d depends only on prices up to d.
* A model used at origin t is trained only on rows whose TARGET month is on
  or before its fit date, and the fit date is never after t.
* Models are refitted every `refit_months` months (default 3) at calendar
  quarter starts; between refits the older model is reused, which can only
  make the backtest look worse, never better.

Shrinkage towards naive (PooledShrunk)
-------------------------------------
forecast = latest price x exp(lambda x predicted log change), with one lambda
per crop and horizon. Lambda is fitted on a separate calibration period
(origins 2015-2021, config.CALIBRATION_START), before validation, so
validation can still choose between models fairly and test stays untouched.
lambda = 0 is exactly naive; lambda = 1 is the raw model. One lambda per
horizon is shared by all crops (per-crop values were too noisy).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg

# All columns built by build_features (the forecaster looks rows up by these).
FEATURES = [
    "r1", "r3", "r6", "r12", "vol6", "dev12", "age",
    "gap_nat", "gap_reg", "nat_r1", "nat_r3", "nat_r12", "reg_r3",
    "n_markets", "month_sin", "month_cos", "region_code",
]
# What each model uses; "seas" is added at fit time from the training rows.
TREE_FEATURES = ["gap_nat", "gap_reg", "dev12", "vol6", "age", "nat_r12",
                 "month_sin", "month_cos", "region_code", "seas"]
LINEAR_FEATURES = ["gap_nat", "gap_reg", "dev12", "r1", "r3", "r12", "nat_r3",
                   "nat_r12", "month_sin", "month_cos", "seas"]

GBM_PARAMS = dict(
    n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
    objective="reg:absoluteerror", tree_method="hist", n_jobs=4, random_state=42,
)
TARGET_CLIP = 1.0         # train on log changes clipped to +/-1 (a 2.7x move)
SEASONAL_PRIOR = 10       # rows of weight given to the crop-wide seasonal mean

CARRY_MONTHS = 2          # carry a real price forward this many months, no more
MIN_TRAIN_ROWS = 150      # per crop and horizon


# -----------------------------------------------------------------------------
# Features (causal: the row for month d only uses prices up to d)
# -----------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE]
    p = p[["commodity", "market", "region", "date", "price_observed"]].copy()
    p = p.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
    g = p.groupby(["commodity", "market"], sort=False)

    p["lp_obs"] = np.log(p["price_observed"])
    p["lp"] = g["lp_obs"].ffill(limit=CARRY_MONTHS)
    # months since the last real price (0 = observed this month)
    obs = p["lp_obs"].notna()
    run = obs.groupby([p["commodity"], p["market"]]).cumsum()
    p["age"] = p.groupby([p["commodity"], p["market"], run]).cumcount()
    p.loc[run == 0, "age"] = np.nan

    lp = g["lp"]
    for k in (1, 3, 6, 12):
        p[f"r{k}"] = p["lp"] - lp.shift(k)
    g = p.groupby(["commodity", "market"], sort=False)
    p["vol6"] = g["r1"].transform(lambda s: s.rolling(6, min_periods=3).std())
    p["dev12"] = p["lp"] - lp.transform(lambda s: s.rolling(12, min_periods=6).mean())

    nat = (p.groupby(["commodity", "date"])["lp"]
           .agg(nat="mean", n_markets="count").reset_index())
    ng = nat.groupby("commodity")["nat"]
    for k in (1, 3, 12):
        nat[f"nat_r{k}"] = nat["nat"] - ng.shift(k)
    p = p.merge(nat, on=["commodity", "date"], how="left")
    p["gap_nat"] = p["lp"] - p["nat"]

    reg = p.groupby(["commodity", "region", "date"])["lp"].mean().rename("reg").reset_index()
    reg["reg_r3"] = reg["reg"] - reg.groupby(["commodity", "region"])["reg"].shift(3)
    p = p.merge(reg, on=["commodity", "region", "date"], how="left")
    p["gap_reg"] = p["lp"] - p["reg"]

    m = p["date"].dt.month
    p["month"] = m
    p["month_sin"] = np.sin(2 * np.pi * m / 12)
    p["month_cos"] = np.cos(2 * np.pi * m / 12)
    p["region_code"] = p["region"].map({r: i for i, r in enumerate(cfg.REGIONS)}).astype(float)

    # Targets: real price h months later vs the real price now (origin observed).
    p = p.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
    g = p.groupby(["commodity", "market"], sort=False)["lp_obs"]
    for h in cfg.HORIZONS:
        p[f"y{h}"] = g.shift(-h) - p["lp_obs"]
        p[f"tdate{h}"] = p["date"] + pd.DateOffset(months=h)
    return p.set_index(["commodity", "market", "date"]).sort_index()


def fit_date_for(origin: pd.Timestamp, refit_months: int) -> pd.Timestamp:
    """The most recent refit date on or before the origin (quarter starts by default)."""
    origin = pd.Timestamp(origin)
    if refit_months <= 1:
        return origin
    month0 = (origin.month - 1) // refit_months * refit_months + 1
    return pd.Timestamp(year=origin.year, month=month0, day=1)


# -----------------------------------------------------------------------------
# Forecasters
# -----------------------------------------------------------------------------
class _Ensemble:
    """Average of a small XGBoost model and a Huber linear model, with the
    seasonal encoding learnt from the same training rows."""

    def __init__(self, params: dict):
        self.params = params

    def _seas(self, d: pd.DataFrame) -> np.ndarray:
        a = self.crop_month.reindex(d["month"]).to_numpy()
        r = self.region_month.reindex(pd.MultiIndex.from_arrays([d["region"], d["month"]]))
        n, m = r["count"].fillna(0).to_numpy(), r["mean"].fillna(0).to_numpy()
        return np.nan_to_num((n * m + SEASONAL_PRIOR * a) / (n + SEASONAL_PRIOR))

    def fit(self, train: pd.DataFrame, y: pd.Series) -> "_Ensemble":
        import xgboost as xgb
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import HuberRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = y.clip(-TARGET_CLIP, TARGET_CLIP)
        self.crop_month = y.groupby(train["month"].to_numpy()).mean()
        self.region_month = (y.groupby([train["region"].to_numpy(), train["month"].to_numpy()])
                             .agg(["mean", "count"]))
        t = train.assign(seas=self._seas(train))
        self.tree = xgb.XGBRegressor(**self.params).fit(t[TREE_FEATURES], y)
        self.linear = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                                    HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500))
        self.linear.fit(t[LINEAR_FEATURES], y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x = x.assign(seas=self._seas(x))
        return (self.tree.predict(x[TREE_FEATURES].astype(float))
                + self.linear.predict(x[LINEAR_FEATURES].astype(float))) / 2


class PooledGBM:
    """Backtest-compatible forecaster: call(history, horizon) -> price forecasts."""

    name = "pooled_gbm"

    def __init__(self, panel: pd.DataFrame, refit_months: int = 3, params: dict | None = None):
        self.feats = build_features(panel)
        self.refit_months = refit_months
        self.params = {**GBM_PARAMS, **(params or {})}
        self._models: dict = {}

    def _model(self, crop: str, h: int, fit_date: pd.Timestamp):
        key = (crop, h, fit_date)
        if key not in self._models:
            try:
                d = self.feats.loc[crop]
            except KeyError:
                self._models[key] = None
                return None
            ok = d[f"y{h}"].notna() & (d[f"tdate{h}"] <= fit_date)
            train = d[ok]
            self._models[key] = (None if len(train) < MIN_TRAIN_ROWS
                                 else _Ensemble(self.params).fit(train, train[f"y{h}"]))
        return self._models[key]

    def predict_log_change(self, crop: str, market: str, origin: pd.Timestamp,
                           horizon: int) -> np.ndarray:
        origin = pd.Timestamp(origin)
        try:
            row = self.feats.loc[(crop, market, origin), FEATURES]
        except KeyError:
            return np.full(horizon, np.nan)
        x = row.to_frame().T.astype(float)
        x["month"] = origin.month
        x["region"] = self.feats.at[(crop, market, origin), "region"]
        fit_date = fit_date_for(origin, self.refit_months)
        out = []
        for h in range(1, horizon + 1):
            m = self._model(crop, h, fit_date)
            out.append(np.nan if m is None else float(m.predict(x)[0]))
        return np.array(out)

    def __call__(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        crop, market = history["commodity"].iloc[0], history["market"].iloc[0]
        origin = history["date"].iloc[-1]
        p0 = history["price_observed"].iloc[-1]
        if not np.isfinite(p0):
            return np.full(horizon, np.nan)
        return p0 * np.exp(self.predict_log_change(crop, market, origin, horizon))


class PooledShrunk:
    """The pooled forecast shrunk towards naive by a per-crop, per-horizon lambda."""

    name = "pooled_shrunk"

    def __init__(self, base: PooledGBM, lambdas: dict):
        self.base = base
        self.lambdas = lambdas

    def lam(self, crop: str, h: int) -> float:
        return float(self.lambdas.get(crop, {}).get(str(h), 1.0))

    def __call__(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        crop, market = history["commodity"].iloc[0], history["market"].iloc[0]
        p0 = history["price_observed"].iloc[-1]
        if not np.isfinite(p0):
            return np.full(horizon, np.nan)
        lc = self.base.predict_log_change(crop, market, history["date"].iloc[-1], horizon)
        lam = np.array([self.lam(crop, h) for h in range(1, horizon + 1)])
        return p0 * np.exp(lam * lc)


# -----------------------------------------------------------------------------
# Fitting lambda on the calibration period
# -----------------------------------------------------------------------------
LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)   # never amplify the model


def fit_lambdas(fc: pd.DataFrame, model: str = "pooled_gbm", pool_crops: bool = True) -> dict:
    """
    Choose lambda to minimise total absolute error on the given forecasts (the
    calibration split). With `pool_crops` (the default) one lambda per horizon
    is shared by all crops: per-crop values were tried and were too noisy
    (they swung from 0 to 1.1 between neighbouring horizons). Returns
    {crop: {"h": lambda}} either way.
    """
    f = fc[fc["model"] == model].copy()
    f["lc"] = np.log(f["forecast"] / f["origin_price"])

    def best(d: pd.DataFrame) -> float:
        errs = [np.abs(d["origin_price"] * np.exp(lam * d["lc"]) - d["actual"]).sum()
                for lam in LAMBDA_GRID]
        return float(LAMBDA_GRID[int(np.argmin(errs))])

    out: dict = {}
    if pool_crops:
        per_h = {str(int(h)): best(d) for h, d in f.groupby("horizon")}
        for crop in sorted(f["commodity"].unique()):
            out[crop] = dict(per_h)
    else:
        for (crop, h), d in f.groupby(["commodity", "horizon"]):
            out.setdefault(crop, {})[str(int(h))] = best(d)
    return out


def save_lambdas(lambdas: dict, n_pairs: dict | None = None) -> None:
    cfg.POOLED_LAMBDAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    cfg.POOLED_LAMBDAS_JSON.write_text(json.dumps(
        {"fitted_on": f"calibration origins {cfg.CALIBRATION_START} to {cfg.VALIDATION_START} "
                      f"(exclusive), targets before {cfg.VALIDATION_START}",
         "lambdas": lambdas, "pairs": n_pairs or {}}, indent=2))


def load_lambdas() -> dict:
    if cfg.POOLED_LAMBDAS_JSON.exists():
        return json.loads(cfg.POOLED_LAMBDAS_JSON.read_text())["lambdas"]
    return {}


# -----------------------------------------------------------------------------
# For the app: one shared instance, refitted at each origin it is asked about
# -----------------------------------------------------------------------------
_APP: dict = {}


def app_forecaster(name: str):
    """
    The pooled forecasters as the app uses them. Models are fitted on all
    targets up to the origin month (refit every month) and cached.
    """
    if "base" not in _APP:
        from . import data
        _APP["base"] = PooledGBM(data.load_monthly(), refit_months=1)
    base = _APP["base"]
    if name == "pooled_gbm":
        return base
    if "shrunk" not in _APP:
        _APP["shrunk"] = PooledShrunk(base, load_lambdas())
    return _APP["shrunk"]


def reset_app_cache() -> None:
    _APP.clear()
