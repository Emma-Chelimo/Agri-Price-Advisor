"""
Walk-forward (rolling-origin) backtest for AgriPrice Advisor.

How it works
------------
For every series and every forecast origin month t in the chosen split:
  1. the forecaster sees only the rows up to and including t
  2. it forecasts t+1 ... t+6
  3. each forecast is scored against the price WFP actually recorded in that
     month; interpolated months are never used as targets
then t moves forward one month and the process repeats.

Rules that keep the comparison fair
-----------------------------------
* Origins must be observed months. An interpolated month before an observed
  origin only uses prices up to the origin, so no future data leaks in.
* Validation and test use calendar cutoffs shared by every series
  (config.VALIDATION_START, config.TEST_START). Validation targets that fall
  in the test period are dropped, so tuning never sees test prices.
* All models are scored on the same (series, origin, horizon) pairs. If any
  model cannot forecast a pair, that pair is dropped for every model.

Metrics
-------
MAE, RMSE, MAPE    usual error measures, per horizon
skill              1 - MAE(model) / MAE(naive) on the same pairs; above 0
                   means the model beats "same as last month"
direction_acc      share of pairs where the model called the direction of the
                   price change (up or down from the origin price) correctly;
                   this matters most for sell-or-store advice
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg
from .forecasters import Forecaster

PAIR_KEYS = ["commodity", "market", "origin", "horizon"]


# -----------------------------------------------------------------------------
# Split definition
# -----------------------------------------------------------------------------
def split_bounds(split: str) -> tuple[pd.Timestamp, pd.Timestamp | None, pd.Timestamp | None]:
    """Return (first origin, last origin exclusive, last target exclusive)."""
    val = pd.Period(cfg.VALIDATION_START, "M").to_timestamp()
    test = pd.Period(cfg.TEST_START, "M").to_timestamp()
    if split == "validation":
        return val, test, test
    if split == "test":
        return test, None, None
    raise ValueError("split must be 'validation' or 'test'")


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
def backtest(
    panel: pd.DataFrame,
    forecasters: dict[str, Forecaster],
    split: str = "validation",
    horizons: Iterable[int] = cfg.HORIZONS,
    series: Iterable[tuple[str, str]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run every forecaster over every origin in the split.

    Returns one row per (model, series, origin, horizon) with the forecast,
    the actual price and the price at the origin.
    """
    horizons = sorted(horizons)
    h_max = max(horizons)
    first_origin, origin_end, target_end = split_bounds(split)
    panel = panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE]
    groups = panel.groupby(["commodity", "market"], sort=True)
    keys = list(series) if series is not None else list(groups.groups.keys())

    rows = []
    t0 = time.time()
    for k, key in enumerate(keys):
        if key not in groups.groups:
            continue
        s = groups.get_group(key).sort_values("date").reset_index(drop=True)
        observed = s["price_observed"].notna().to_numpy()
        n_obs_so_far = np.cumsum(observed)
        dates = s["date"].to_numpy()
        actual = s["price_observed"].to_numpy()

        for i in range(len(s)):
            d = s.at[i, "date"]
            if not observed[i] or d < first_origin:
                continue
            if origin_end is not None and d >= origin_end:
                break
            if n_obs_so_far[i] < cfg.MIN_TRAIN_MONTHS:
                continue
            targets = []
            for h in horizons:
                j = i + h
                if j >= len(s) or not observed[j]:
                    continue
                if target_end is not None and dates[j] >= np.datetime64(target_end):
                    continue
                targets.append((h, j))
            if not targets:
                continue

            history = s.iloc[: i + 1]
            for name, fn in forecasters.items():
                fc = np.asarray(fn(history, h_max), dtype=float)
                for h, j in targets:
                    rows.append((split, name, key[0], key[1], d, h,
                                 s.at[j, "date"], actual[i], actual[j], fc[h - 1]))
        if verbose and (k + 1) % 20 == 0:
            print(f"  [{split}] {k + 1}/{len(keys)} series, {time.time() - t0:.0f}s")

    out = pd.DataFrame(rows, columns=[
        "split", "model", "commodity", "market", "origin", "horizon",
        "target_date", "origin_price", "actual", "forecast",
    ])
    return common_pairs(out)


def common_pairs(fc: pd.DataFrame) -> pd.DataFrame:
    """Keep only pairs that every model forecast (no NaN, finite, > 0)."""
    ok = np.isfinite(fc["forecast"]) & (fc["forecast"] > 0)
    n_models = fc["model"].nunique()
    good = fc[ok].groupby(PAIR_KEYS)["model"].transform("nunique") == n_models
    return fc[ok][good].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _add_errors(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["error"] = fc["forecast"] - fc["actual"]
    fc["abs_error"] = fc["error"].abs()
    fc["sq_error"] = fc["error"] ** 2
    fc["ape"] = fc["abs_error"] / fc["actual"] * 100
    actual_move = np.sign(fc["actual"] - fc["origin_price"])
    pred_move = np.sign(fc["forecast"] - fc["origin_price"])
    # Direction is only scored where the price actually moved and the model
    # actually predicted a move (naive never does, so it has no score).
    fc["dir_scored"] = (actual_move != 0) & (pred_move != 0)
    fc["dir_correct"] = fc["dir_scored"] & (actual_move == pred_move)
    return fc


def series_metrics(fc: pd.DataFrame) -> pd.DataFrame:
    """Error measures per split, model, series and horizon, with skill vs naive."""
    fc = _add_errors(fc)
    g = fc.groupby(["split", "model", "commodity", "market", "horizon"])
    m = g.agg(n=("abs_error", "size"), MAE=("abs_error", "mean"),
              MSE=("sq_error", "mean"), MAPE=("ape", "mean"),
              dir_n=("dir_scored", "sum"), dir_hits=("dir_correct", "sum")).reset_index()
    m["RMSE"] = np.sqrt(m.pop("MSE"))
    m["direction_acc"] = np.where(m["dir_n"] > 0, m["dir_hits"] / m["dir_n"].clip(lower=1), np.nan)
    naive = m[m["model"] == "naive"][["split", "commodity", "market", "horizon", "MAE"]]
    m = m.merge(naive.rename(columns={"MAE": "MAE_naive"}),
                on=["split", "commodity", "market", "horizon"], how="left")
    m["skill"] = 1 - m["MAE"] / m["MAE_naive"]
    return m.drop(columns=["dir_hits"]).round(4)


def summary(fc: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """
    Headline table: one row per split, model and horizon (plus `by`).

    pooled_skill  1 - total absolute error / naive's total absolute error
    median_skill  median of the per-series skill scores
    pct_beat      share of series where the model's MAE is below naive's
    """
    by = by or []
    sm = series_metrics(fc)
    fc = _add_errors(fc)
    fc["series"] = fc["commodity"] + " | " + fc["market"]
    keys = ["split"] + by + ["model", "horizon"]
    naive = (fc[fc["model"] == "naive"]
             .set_index(["split"] + PAIR_KEYS)["abs_error"].rename("abs_error_naive"))
    fc = fc.join(naive, on=["split"] + PAIR_KEYS)
    sm_g = sm.groupby(keys)

    g = fc.groupby(keys)
    out = g.agg(n_pairs=("abs_error", "size"),
                n_series=("series", "nunique"),
                MAE=("abs_error", "mean"), MAPE=("ape", "mean"),
                abs_err=("abs_error", "sum"), abs_err_naive=("abs_error_naive", "sum"),
                dir_n=("dir_scored", "sum"), dir_hits=("dir_correct", "sum")).reset_index()
    out["pooled_skill"] = 1 - out.pop("abs_err") / out.pop("abs_err_naive")
    out["direction_acc"] = np.where(out["dir_n"] > 0,
                                    out.pop("dir_hits") / out["dir_n"].clip(lower=1), np.nan)
    out = out.drop(columns=["dir_n"])
    med = sm_g.agg(median_MAPE=("MAPE", "median"), median_skill=("skill", "median"),
                   pct_beat=("skill", lambda s: 100 * (s > 0).mean())).reset_index()
    out = out.merge(med, on=keys)
    return out.round(3)


def wide_skill(summ: pd.DataFrame, value: str = "pooled_skill") -> pd.DataFrame:
    """Models as rows, horizons as columns; handy for the report."""
    return summ.pivot_table(index=["split", "model"], columns="horizon", values=value).round(3)


# -----------------------------------------------------------------------------
# Run and save
# -----------------------------------------------------------------------------
def run_and_save(panel: pd.DataFrame, forecasters: dict[str, Forecaster],
                 splits=("validation", "test"), out_dir=None, **kw) -> dict:
    out_dir = out_dir or cfg.BACKTEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fcs = [backtest(panel, forecasters, split=s, **kw) for s in splits]
    fc = pd.concat(fcs, ignore_index=True)
    res = {
        "forecasts": fc,
        "series_metrics": series_metrics(fc),
        "summary": summary(fc),
        "summary_by_crop": summary(fc, by=["commodity"]),
    }
    fc.to_parquet(out_dir / "forecasts.parquet", index=False)
    res["series_metrics"].to_csv(out_dir / "metrics_by_series.csv", index=False)
    res["summary"].to_csv(out_dir / "summary.csv", index=False)
    res["summary_by_crop"].to_csv(out_dir / "summary_by_crop.csv", index=False)
    return res
