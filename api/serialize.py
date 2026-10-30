"""Turn back-end objects (dataclasses, pandas, numpy) into plain JSON."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from src import config as cfg
from src.advice_service import AdviceResult


def clean(x):
    """Recursively convert to JSON-safe values; NaN and infinity become None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NaT:
        return None
    return x


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict(orient="records"))


def advice(r: AdviceResult) -> dict:
    fc, adv, mc = r.fc, r.adv, r.mc
    history = r.series[["date", "price", "price_observed"]].copy()
    out = {
        "crop": r.crop, "market": r.market, "quantity_kg": r.quantity_kg, "horizon": r.horizon,
        "latest": {"price": fc.origin_price, "date": fc.origin_date,
                   "months_old": r.months_old, "is_stale": r.is_stale},
        "history": [{"date": d, "price": p, "observed": bool(pd.notna(o))}
                    for d, p, o in zip(history["date"], history["price"], history["price_observed"])],
        "forecast": {
            "method": fc.model, "label": fc.label, "level": fc.level, "note": fc.note,
            "used_fallback": fc.used_fallback,
            "points": [{"date": d, "price": m, "low": lo, "high": hi, "months_ahead": i + 1,
                        "typical_error_pct": e}
                       for i, (d, m, lo, hi, e) in enumerate(zip(
                           fc.dates, fc.median, fc.lower, fc.upper,
                           fc.typical_error_pct or [None] * len(fc.dates)))],
        },
        "costs": {"storage_ugx_per_kg_month": r.costs.storage_ugx_per_kg_month,
                  "loss_pct_per_month": r.costs.loss_rate_per_month * 100,
                  "transport_ugx_per_kg_km": r.costs.transport_ugx_per_kg_km},
        "advice": {
            "recommendation": adv.recommendation, "months": adv.months,
            "expected_net_ugx": adv.expected_net_ugx, "net_sell_now_ugx": adv.net_sell_now_ugx,
            "gain_ugx": adv.gain_ugx, "confidence": adv.confidence,
            "confidence_label": adv.confidence_label,
            "best_store_months": adv.best_store_months, "best_store_prob": adv.best_store_prob,
            "store_min_probability": cfg.STORE_MIN_PROBABILITY,
            "options": records(adv.options),
        },
        "comparison": {
            "reference_month": mc.reference_month, "excluded": mc.excluded,
            "markets": records(mc.table),
            "best_alternative": None if r.alt is None else clean(r.alt.to_dict()),
            "go_to_alternative": r.go_to_alternative,
        },
        "sell_market": r.sell_market,
        "explanation_template": r.template_explanation(),
    }
    return clean(out)
