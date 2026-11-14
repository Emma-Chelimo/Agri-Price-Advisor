"""
SARIMA + residual-learning hybrid (the Group 6 proposed architecture).

    dataset --> SARIMA (seasonal baseline) --> linear price forecast
                         |
                         v
             residuals = actual - SARIMA   (out-of-sample, per horizon)
               |                         |
               v                         v
     LSTM on the sequence of     XGBoost on tabular features
     recent SARIMA errors        (pooled features + SARIMA's forecast)
               \\                        /
                average residual prediction
                          |
                          v
     final forecast = SARIMA forecast x exp(average residual)

Everything is on the log scale, so a "residual" is log(actual / SARIMA), a
percentage-like error that means the same for a 500 UGX and a 5,000 UGX crop.

No look-ahead
-------------
* SARIMA parameters are re-estimated every quarter on prices up to the
  quarter start, then re-filtered (no re-estimation) with prices up to each
  origin. Only the latest unbroken segment of the series is used.
* The residuals the LSTM and XGBoost learn from are genuine out-of-sample
  SARIMA errors: the forecast made at an earlier origin minus the price that
  later arrived. A residual model fitted at date F only sees residuals whose
  TARGET month is on or before F.
* Residual models are refitted at calendar quarter starts, like the pooled model.
"""

from __future__ import annotations

import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as cfg

H = max(cfg.HORIZONS)
SARIMA_ORDERS = {
    "A": dict(order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), trend="n"),
    "B": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend="n"),   # "airline"
    "C": dict(order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), trend="t"),   # with drift
}
MIN_SEASONAL = 36      # months in the segment needed for the seasonal model
MIN_ARIMA = 12         # below this, fall back to naive (flat) forecast


# -----------------------------------------------------------------------------
# SARIMA table: forecasts for every observed origin of every series
# -----------------------------------------------------------------------------
def _quarter_start(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=d.year, month=(d.month - 1) // 3 * 3 + 1, day=1)


def _series_sarima(args) -> list[tuple]:
    key, s, spec, first_origin = args
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    s = s.sort_values("date").reset_index(drop=True)
    observed = s["price_observed"].notna().to_numpy()
    n_obs = np.cumsum(observed)
    rows, cache = [], {}
    for i in range(len(s)):
        d = s.at[i, "date"]
        if not observed[i] or n_obs[i] < cfg.MIN_TRAIN_MONTHS or d < first_origin:
            continue
        seg = s.iloc[: i + 1]
        seg = seg[seg["segment_id"] == seg["segment_id"].iloc[-1]]
        y = np.log(seg.set_index("date")["price"].dropna())
        y.index = pd.DatetimeIndex(y.index, freq="MS")
        p0 = float(s.at[i, "price_observed"])
        fq = _quarter_start(d)
        y_fit = y[y.index <= fq]
        kind = "flat"
        lc = np.zeros(H)
        if len(y_fit) >= MIN_ARIMA:
            kind = "seasonal" if len(y_fit) >= MIN_SEASONAL else "arima"
            ck = (kind, fq, y_fit.index[0])
            if ck not in cache:
                kw = dict(spec) if kind == "seasonal" else dict(order=(1, 1, 0), trend=spec["trend"])
                try:
                    cache[ck] = SARIMAX(y_fit, **kw).fit(disp=False)
                except Exception:
                    cache[ck] = None
            res = cache[ck]
            if res is not None:
                try:
                    fc = res.apply(y).forecast(H).to_numpy()
                    lc = fc - np.log(p0)
                    if not np.all(np.isfinite(lc)) or np.abs(lc).max() > 1.5:
                        lc, kind = np.zeros(H), "flat"
                except Exception:
                    lc, kind = np.zeros(H), "flat"
            else:
                kind = "flat"
        rows.append((key[0], key[1], d, p0, kind, *lc))
    return rows


def sarima_table(panel: pd.DataFrame, order: str = "A", first_origin: str | None = None,
                 processes: int | None = None) -> pd.DataFrame:
    """One row per (series, observed origin): SARIMA log change for h = 1..6."""
    p = panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE]
    fo = pd.Timestamp(first_origin) if first_origin else pd.Timestamp("1900-01-01")
    jobs = [(k, g[["date", "price", "price_observed", "segment_id"]], SARIMA_ORDERS[order], fo)
            for k, g in p.groupby(["commodity", "market"])]
    with Pool(processes or os.cpu_count()) as pool:
        out = pool.map(_series_sarima, jobs, chunksize=2)
    cols = ["commodity", "market", "origin", "origin_price", "kind"] + [f"sar{h}" for h in range(1, H + 1)]
    return pd.DataFrame([r for rows in out for r in rows], columns=cols)


def attach_residuals(sar: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add the observed log price h months later and the SARIMA residual for each h."""
    p = panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE][["commodity", "market", "date", "price_observed"]]
    lp = p.assign(lp=np.log(p["price_observed"])).set_index(["commodity", "market", "date"])["lp"]
    out = sar.copy()
    for h in range(1, H + 1):
        tgt = out["origin"] + pd.DateOffset(months=h)
        idx = pd.MultiIndex.from_arrays([out["commodity"], out["market"], tgt])
        out[f"tdate{h}"] = tgt
        out[f"real{h}"] = lp.reindex(idx).to_numpy() - np.log(out["origin_price"])
        out[f"res{h}"] = out[f"real{h}"] - out[f"sar{h}"]
    return out


# -----------------------------------------------------------------------------
# Residual models
# -----------------------------------------------------------------------------
RES_CLIP = 0.7
SEQ_LEN = 12
CROPS = None   # filled from data


def _seq_inputs(t: pd.DataFrame, feats: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    LSTM inputs: for each origin, the last SEQ_LEN months of
      * the one-step SARIMA error known by then (error of the forecast made a
        month earlier for this month), 0 where unknown, plus a known/unknown flag
      * the month-on-month log change (carried prices, from pooled features)
      * month sin/cos
    and constant crop one-hot channels. Shape (n, SEQ_LEN, channels).
    """
    e1 = t.set_index(["commodity", "market", "tdate1"])["res1"].clip(-RES_CLIP, RES_CLIP)
    e1.index = e1.index.set_names(["commodity", "market", "date"])
    e1 = e1[~e1.index.duplicated()]
    r1 = feats["r1"]
    crops = sorted(t["commodity"].unique()) if CROPS is None else CROPS
    n = len(t)
    X = np.zeros((n, SEQ_LEN, 5 + len(crops)), dtype=np.float32)
    for lag in range(SEQ_LEN):
        d = t["origin"] - pd.DateOffset(months=SEQ_LEN - 1 - lag)
        idx = pd.MultiIndex.from_arrays([t["commodity"], t["market"], d])
        ev = e1.reindex(idx).to_numpy()
        # the one-step error for month d is known at origin only if d <= origin (always true here)
        X[:, lag, 0] = np.nan_to_num(ev)
        X[:, lag, 1] = np.isfinite(ev)
        X[:, lag, 2] = np.nan_to_num(r1.reindex(idx).to_numpy()).clip(-0.5, 0.5)
        X[:, lag, 3] = np.sin(2 * np.pi * d.dt.month / 12)
        X[:, lag, 4] = np.cos(2 * np.pi * d.dt.month / 12)
    for j, c in enumerate(crops):
        X[:, :, 5 + j] = (t["commodity"].to_numpy() == c)[:, None]
    return X


def _masked_huber(y_true, y_pred):
    from keras import ops
    mask = ops.logical_not(ops.isnan(y_true))
    yt = ops.where(mask, y_true, 0.0)
    d = ops.abs(yt - y_pred)
    hub = ops.where(d < 0.1, 0.5 * d * d, 0.1 * (d - 0.05))
    m = ops.cast(mask, "float32")
    return ops.sum(hub * m) / ops.maximum(ops.sum(m), 1.0)


class LSTMResidual:
    """One small LSTM pooled over all crops, outputs residuals for h = 1..6."""

    def __init__(self, units: int = 16, epochs: int = 40, seeds=(0, 1, 2)):
        self.units, self.epochs, self.seeds = units, epochs, seeds

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LSTMResidual":
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras
        self.models = []
        n_val = max(1, int(0.15 * len(X)))     # last 15% (chronological) for early stopping
        for seed in self.seeds:
            keras.utils.set_random_seed(seed)
            m = keras.Sequential([
                keras.Input(X.shape[1:]),
                keras.layers.LSTM(self.units, dropout=0.1),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(H),
            ])
            m.compile(optimizer=keras.optimizers.Adam(3e-3), loss=_masked_huber)
            m.fit(X[:-n_val], Y[:-n_val], validation_data=(X[-n_val:], Y[-n_val:]),
                  epochs=self.epochs, batch_size=128, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
            self.models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X, verbose=0, batch_size=1024) for m in self.models], axis=0)


XGB_RES_FEATURES = ["gap_nat", "gap_reg", "dev12", "vol6", "age", "nat_r12", "r3", "r12",
                    "month_sin", "month_cos", "region_code", "sar", "e1_mean3", "e1_last"]


def _xgb_frame(t: pd.DataFrame, feats: pd.DataFrame, X_seq: np.ndarray, h: int) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([t["commodity"], t["market"], t["origin"]])
    f = feats.reindex(idx)[["gap_nat", "gap_reg", "dev12", "vol6", "age", "nat_r12", "r3", "r12",
                             "month_sin", "month_cos", "region_code"]].reset_index(drop=True)
    f["sar"] = t[f"sar{h}"].to_numpy()
    known = X_seq[:, -3:, 1].sum(axis=1)
    f["e1_mean3"] = np.where(known > 0, X_seq[:, -3:, 0].sum(axis=1) / np.maximum(known, 1), np.nan)
    f["e1_last"] = np.where(X_seq[:, -1, 1] > 0, X_seq[:, -1, 0], np.nan)
    return f


def run_residual_models(t: pd.DataFrame, feats: pd.DataFrame, fit_dates: list[pd.Timestamp],
                        use_lstm: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    For each quarterly fit date F, train the LSTM (pooled) and XGBoost (per crop and
    horizon) on residuals with target month <= F, then predict residuals for the
    origins in [F, next F). Returns t with columns lstm1..6 and xgb1..6.
    """
    import time

    import xgboost as xgb

    from .pooled import GBM_PARAMS
    global CROPS
    CROPS = sorted(t["commodity"].unique())
    t = t.sort_values(["origin", "commodity", "market"]).reset_index(drop=True)
    X_seq = _seq_inputs(t, feats)
    Y = np.stack([t[f"res{h}"].clip(-RES_CLIP, RES_CLIP).to_numpy() for h in range(1, H + 1)], axis=1)
    tdates = np.stack([t[f"tdate{h}"].to_numpy() for h in range(1, H + 1)], axis=1)
    frames = {h: _xgb_frame(t, feats, X_seq, h) for h in range(1, H + 1)}
    for h in range(1, H + 1):
        t[f"lstm{h}"] = np.nan
        t[f"xgb{h}"] = np.nan

    t0 = time.time()
    fit_dates = sorted(fit_dates)
    for k, F in enumerate(fit_dates):
        F64 = np.datetime64(F)
        nxt = np.datetime64(fit_dates[k + 1]) if k + 1 < len(fit_dates) else np.datetime64("2100-01-01")
        pred_rows = ((t["origin"] >= F) & (t["origin"] < nxt)).to_numpy()
        if not pred_rows.any():
            continue
        Ytr = np.where(tdates <= F64, Y, np.nan)
        tr = ~np.all(np.isnan(Ytr), axis=1) & (t["origin"].to_numpy() < F64)
        if use_lstm and tr.sum() >= 300:
            lm = LSTMResidual().fit(X_seq[tr], Ytr[tr])
            pr = lm.predict(X_seq[pred_rows])
            for h in range(1, H + 1):
                t.loc[pred_rows, f"lstm{h}"] = pr[:, h - 1]
        for h in range(1, H + 1):
            for crop in CROPS:
                c = (t["commodity"] == crop).to_numpy()
                trc = tr & c & ~np.isnan(Ytr[:, h - 1])
                pc = pred_rows & c
                if not pc.any() or trc.sum() < 100:
                    continue
                m = xgb.XGBRegressor(**{**GBM_PARAMS, "n_jobs": 2})
                m.fit(frames[h][trc], Ytr[trc, h - 1])
                t.loc[pc, f"xgb{h}"] = m.predict(frames[h][pc])
        if verbose:
            print(f"  fit {F.date()}: train rows {int(tr.sum())}, predict {int(pred_rows.sum())}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return t


def to_forecasts(t: pd.DataFrame) -> pd.DataFrame:
    """Long table of price forecasts for each hybrid variant."""
    rows = []
    for h in range(1, H + 1):
        sar = t[f"sar{h}"]
        lstm = t[f"lstm{h}"].fillna(0.0)
        xg = t[f"xgb{h}"].fillna(0.0)
        variants = {
            "sarima": sar,
            "sarima_lstm": sar + lstm,
            "sarima_xgb": sar + xg,
            "hybrid": sar + (lstm + xg) / 2,
        }
        for name, lc in variants.items():
            rows.append(pd.DataFrame({
                "model": name, "commodity": t["commodity"], "market": t["market"],
                "origin": t["origin"], "horizon": h, "forecast": t["origin_price"] * np.exp(lc),
            }))
    return pd.concat(rows, ignore_index=True)
