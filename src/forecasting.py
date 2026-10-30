"""
Forecast serving for the app (Phase 4).

Two parts:

1. build_registry()  reads the walk-forward backtest, picks one forecasting
                     method per crop using the VALIDATION split only, and
                     works out forecast bands from that method's past errors.
                     The result is saved to processed/model_registry.json.

2. forecast()        what the app calls. Given a crop, market and horizon it
                     looks up the crop's method in the registry, runs it on
                     the market's price history, and returns the forecast
                     with its band.

Selection rule
--------------
For each crop, average each candidate's validation skill over horizons 1-6.
The best candidate replaces naive only if that average is at least
SELECTION_MIN_SKILL above zero. Otherwise the crop uses naive.

Forecast bands
--------------
For the chosen method, crop and horizon, take r = log(actual / forecast) over
every validation forecast. The band half-width w is the 80th percentile of
|r|, so 80% of past forecasts were within a factor exp(w) of the real price.
The band is [f / exp(w), f * exp(w)], centred on the forecast. Centring keeps
the band, the forecast line and the decision probabilities consistent with
one another (the forecast is the middle of the range). The test split is used
only to check how often real prices landed inside the band.

For decisions, the price k months ahead is treated as log-normal around the
forecast with sigma = w / z, where z is the normal quantile for the band level
(1.2816 for 80%).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as pipeline
from . import evaluation as ev
from . import forecasters as F


# -----------------------------------------------------------------------------
# 1. Build the registry from backtest results
# -----------------------------------------------------------------------------
def band_z(level: float) -> float:
    """Normal quantile for a central band, e.g. 1.2816 for 80%."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + level / 2)


def _log_ratio_quantiles(fc: pd.DataFrame, level: float) -> tuple[float, float, int]:
    """Symmetric band: half-width is the `level` quantile of |log error|."""
    w = float(np.log(fc["actual"] / fc["forecast"]).abs().quantile(level))
    return -w, w, int(len(fc))


def _coverage(fc: pd.DataFrame, q_lo: float, q_hi: float) -> float | None:
    if fc.empty:
        return None
    inside = (fc["actual"] >= fc["forecast"] * np.exp(q_lo)) & \
             (fc["actual"] <= fc["forecast"] * np.exp(q_hi))
    return round(float(inside.mean()), 3)


def build_registry(forecasts: pd.DataFrame | None = None, write: bool = True) -> dict:
    if forecasts is None:
        forecasts = pd.read_parquet(cfg.BACKTEST_DIR / "forecasts.parquet")
    val = forecasts[forecasts["split"] == "validation"]
    test = forecasts[forecasts["split"] == "test"]
    if val.empty:
        raise ValueError("No validation forecasts found. Run scripts/run_backtest.py first.")

    by_crop = ev.summary(val, by=["commodity"])
    by_crop = by_crop[~by_crop["model"].isin(cfg.NOT_SELECTABLE)]
    level = cfg.INTERVAL_LEVEL

    crops = {}
    for crop in cfg.TARGET_CROPS:
        s = by_crop[by_crop["commodity"] == crop]
        if s.empty:
            choice, mean_skill, reason = cfg.FALLBACK_MODEL, None, "no validation results for this crop"
        else:
            avg = s.groupby("model")["pooled_skill"].mean().sort_values(ascending=False)
            best, best_skill = avg.index[0], float(avg.iloc[0])
            if best != cfg.FALLBACK_MODEL and best_skill >= cfg.SELECTION_MIN_SKILL:
                choice, mean_skill = best, best_skill
                reason = (f"best average validation skill ({best_skill:+.3f}), "
                          f"above the {cfg.SELECTION_MIN_SKILL:+.2f} margin")
            else:
                choice, mean_skill = cfg.FALLBACK_MODEL, 0.0
                challenger = avg.drop(cfg.FALLBACK_MODEL, errors="ignore")
                if len(challenger):
                    reason = (f"no method beat naive on validation; best challenger was "
                              f"{challenger.index[0]} at {float(challenger.iloc[0]):+.3f}")
                else:
                    reason = "no challengers evaluated"

        horizons = {}
        for h in cfg.HORIZONS:
            v = val[(val["model"] == choice) & (val["commodity"] == crop) & (val["horizon"] == h)]
            source = "crop"
            if len(v) < cfg.INTERVAL_MIN_PAIRS:
                v = val[(val["model"] == choice) & (val["horizon"] == h)]
                source = "all crops"
            q_lo, q_hi, n = _log_ratio_quantiles(v, level)
            t = test[(test["model"] == choice) & (test["commodity"] == crop) & (test["horizon"] == h)]
            row = s[(s["model"] == choice) & (s["horizon"] == h)] if not s.empty else s
            horizons[str(h)] = {
                "validation_MAPE": round(float(row["MAPE"].iloc[0]), 2) if len(row) else None,
                "validation_skill": round(float(row["pooled_skill"].iloc[0]), 3) if len(row) else None,
                "band_log_low": round(q_lo, 4),
                "band_log_high": round(q_hi, 4),
                "log_sigma": round(q_hi / band_z(level), 4),
                "band_pairs": n,
                "band_source": source,
                "test_band_coverage": _coverage(t, q_lo, q_hi),
                "test_pairs": int(len(t)),
            }

        crops[crop] = {
            "model": choice,
            "label": F.MODEL_LABELS.get(choice, choice),
            "mean_validation_skill": None if mean_skill is None else round(mean_skill, 3),
            "reason": reason,
            "horizons": horizons,
        }

    registry = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "selection_split": "validation",
        "validation_window": f"origins {cfg.VALIDATION_START} to {cfg.TEST_START} (exclusive)",
        "selection_rule": (f"highest mean validation skill over horizons "
                           f"{cfg.HORIZONS[0]}-{cfg.HORIZONS[-1]}, must exceed "
                           f"{cfg.SELECTION_MIN_SKILL} to replace {cfg.FALLBACK_MODEL}"),
        "candidates": sorted(set(val["model"]) - cfg.NOT_SELECTABLE),
        "interval_level": level,
        "max_horizon": cfg.MAX_FORECAST_HORIZON,
        "crops": crops,
    }
    if write:
        cfg.MODEL_REGISTRY_JSON.parent.mkdir(parents=True, exist_ok=True)
        cfg.MODEL_REGISTRY_JSON.write_text(json.dumps(registry, indent=2))
    return registry


def load_registry() -> dict:
    """Read the registry. If it is missing, every crop falls back to naive."""
    if cfg.MODEL_REGISTRY_JSON.exists():
        return json.loads(cfg.MODEL_REGISTRY_JSON.read_text())
    return {"crops": {}, "interval_level": cfg.INTERVAL_LEVEL,
            "max_horizon": cfg.MAX_FORECAST_HORIZON, "missing": True}


# -----------------------------------------------------------------------------
# 2. Serve a forecast
# -----------------------------------------------------------------------------
@dataclass
class Forecast:
    crop: str
    market: str
    model: str
    label: str
    origin_date: pd.Timestamp          # month of the latest observed price
    origin_price: float
    dates: list[pd.Timestamp]
    median: list[float]
    lower: list[float]
    upper: list[float]
    level: float
    log_sigma: list[float] = field(default_factory=list)
    typical_error_pct: list[float | None] = field(default_factory=list)
    used_fallback: bool = False
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates, "forecast": self.median,
            "lower": self.lower, "upper": self.upper,
            "months_ahead": range(1, len(self.dates) + 1),
            "typical_error_pct": self.typical_error_pct,
        })


def forecast(panel: pd.DataFrame, crop: str, market: str, horizon: int,
             registry: dict | None = None) -> Forecast:
    """
    Forecast `horizon` months after the latest observed price for this series.

    Uses the crop's method from the registry. If that method cannot produce a
    forecast for this market (for example XGBoost on a short history), naive
    is used instead and `used_fallback` is set.
    """
    series = pipeline.get_series(panel, crop, market)
    return forecast_series(series, crop, market, horizon, registry)


def forecast_series(series: pd.DataFrame, crop: str, market: str, horizon: int,
                    registry: dict | None = None) -> Forecast:
    """
    Same as forecast(), for one series that has already been selected (sorted
    by date). The walk-forward advice evaluation passes truncated histories
    here, so nothing after the decision month is ever seen.
    """
    registry = registry or load_registry()
    horizon = int(max(1, min(horizon, registry.get("max_horizon", cfg.MAX_FORECAST_HORIZON))))

    if series.empty or series["price_observed"].notna().sum() == 0:
        raise ValueError(f"No price history for {crop} at {market}.")
    # The forecast starts from the latest month WFP actually recorded.
    last_obs = series.index[series["price_observed"].notna()][-1]
    history = series.loc[:last_obs]

    entry = registry.get("crops", {}).get(crop)
    model = entry["model"] if entry else cfg.FALLBACK_MODEL
    note = "" if entry else "No model registry found, so the simple naive forecast is used."

    used_fallback = False
    values = np.asarray(F.get_forecaster(model)(history, horizon), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        used_fallback = model != cfg.FALLBACK_MODEL
        note = (f"{F.MODEL_LABELS.get(model, model)} could not forecast this market "
                f"(history too short or broken by gaps), so naive is used instead.")
        model = cfg.FALLBACK_MODEL
        values = np.asarray(F.naive(history, horizon), dtype=float)

    # Bands: from the registry entry if it describes the model actually used.
    lows, highs, errs, sigmas = [], [], [], []
    hz = entry["horizons"] if entry and entry["model"] == model else {}
    for h in range(1, horizon + 1):
        b = hz.get(str(h))
        if b:
            lows.append(b["band_log_low"]); highs.append(b["band_log_high"])
            errs.append(b["validation_MAPE"])
            sigmas.append(b.get("log_sigma", b["band_log_high"] / band_z(registry.get("interval_level", cfg.INTERVAL_LEVEL))))
        else:
            lows.append(np.nan); highs.append(np.nan); errs.append(None); sigmas.append(np.nan)
    lows, highs = np.array(lows, dtype=float), np.array(highs, dtype=float)

    origin_date = pd.Timestamp(history["date"].iloc[-1])
    return Forecast(
        crop=crop, market=market, model=model,
        label=F.MODEL_LABELS.get(model, model),
        origin_date=origin_date,
        origin_price=float(history["price_observed"].iloc[-1]),
        dates=list(pd.date_range(origin_date + pd.offsets.MonthBegin(1),
                                 periods=horizon, freq="MS")),
        median=values.round(0).tolist(),
        lower=(values * np.exp(lows)).round(0).tolist(),
        upper=(values * np.exp(highs)).round(0).tolist(),
        level=registry.get("interval_level", cfg.INTERVAL_LEVEL),
        log_sigma=sigmas,
        typical_error_pct=errs,
        used_fallback=used_fallback,
        note=note,
    )
