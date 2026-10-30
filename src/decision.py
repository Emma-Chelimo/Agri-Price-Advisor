"""
Sell-or-store advice and market comparison (Phase 5).

Every number shown on the dashboard comes from a formula in this file.

Notation for a farmer holding q kg at their market, with the latest observed
price p0 and a forecast f_k for k months ahead:

    surviving_k   = q * (1 - loss) ** k                   kg left after k months
    net_now       = p0 * q                                 sell today
    net_k         = f_k * surviving_k - storage * q * k    store k months, then sell
    gain_k        = net_k - net_now

At another market m, d_m km away by road, the transport cost is
    transport_m   = rate * d_m * q
and is subtracted from every option at that market. At the home market it is
zero; getting produce to the home market costs the same whichever month the
farmer sells, so it cannot change the timing advice and is left out.

Uncertainty
-----------
The price k months ahead is treated as log-normal around f_k with the sigma
from the forecast band (src/forecasting.py). Storing k months beats selling
now when the price beats the break-even price

    b_k = (p0 * q + storage * q * k) / surviving_k

so  P(store k beats selling now) = 1 - Phi( ln(b_k / f_k) / sigma_k ).

Recommendation
--------------
Store for k months if that option has the highest median net and its chance
of beating selling now is at least STORE_MIN_PROBABILITY. Otherwise sell now.

Confidence
----------
For "store k":   P(store k beats selling now).
For "sell now":  P(no storage option beats selling now). Forecast errors at
different horizons move together (a price that ends up high at month 3 is
usually high at month 4 too), so this is taken as 1 - max_k P(store k beats
selling now), which is exact when errors are perfectly correlated.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as pipeline
from . import forecasting

_N = NormalDist()


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
@dataclass
class Costs:
    storage_ugx_per_kg_month: float
    loss_rate_per_month: float          # fraction, e.g. 0.0045
    transport_ugx_per_kg_km: float = cfg.TRANSPORT_UGX_PER_KG_KM

    @classmethod
    def default_for(cls, crop: str) -> "Costs":
        c = cfg.CROP_COSTS.get(crop, {"storage_cost_ugx_per_kg_month": 20,
                                      "storage_loss_pct_per_month": 1.0})
        return cls(storage_ugx_per_kg_month=float(c["storage_cost_ugx_per_kg_month"]),
                   loss_rate_per_month=float(c["storage_loss_pct_per_month"]) / 100)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def road_km(markets: pd.DataFrame, home: str) -> pd.Series:
    """Approximate road distance from `home` to every market, in km."""
    m = markets.set_index("market")
    h = m.loc[home]
    d = haversine_km(h["latitude"], h["longitude"], m["latitude"], m["longitude"])
    return pd.Series(d * cfg.ROAD_FACTOR, index=m.index).round(0)


def confidence_label(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "Not available"
    for threshold, label in cfg.CONFIDENCE_LABELS:
        if p >= threshold:
            return label
    return cfg.CONFIDENCE_LABELS[-1][1]


# -----------------------------------------------------------------------------
# Timing at one market
# -----------------------------------------------------------------------------
def timing_options(fc: forecasting.Forecast, qty: float, costs: Costs,
                   transport_ugx: float = 0.0, start_price: float | None = None) -> pd.DataFrame:
    """
    One row per option: sell now, or store 1..H months then sell at this market.

    `start_price` is the price for selling now (defaults to the forecast's
    origin price). `transport_ugx` is subtracted from every option.
    """
    p0 = fc.origin_price if start_price is None else start_price
    net_now = p0 * qty - transport_ugx
    rows = [{"action": "Sell now", "months": 0, "price": p0, "low": p0, "high": p0,
             "kg_sold": qty, "storage_ugx": 0.0, "transport_ugx": transport_ugx,
             "net_ugx": net_now, "gain_vs_now_ugx": 0.0, "breakeven_price": np.nan,
             "prob_beats_now": np.nan}]
    for k, (f, lo, hi, sig) in enumerate(zip(fc.median, fc.lower, fc.upper, fc.log_sigma), start=1):
        surviving = qty * (1 - costs.loss_rate_per_month) ** k
        storage = costs.storage_ugx_per_kg_month * qty * k
        net = f * surviving - storage - transport_ugx
        breakeven = (p0 * qty + storage) / surviving
        if sig is not None and np.isfinite(sig) and sig > 0:
            prob = 1 - _N.cdf(np.log(breakeven / f) / sig)
        else:
            prob = np.nan
        rows.append({"action": f"Store {k} month{'s' if k > 1 else ''}", "months": k,
                     "price": f, "low": lo, "high": hi, "kg_sold": surviving,
                     "storage_ugx": storage, "transport_ugx": transport_ugx,
                     "net_ugx": net, "gain_vs_now_ugx": net - net_now,
                     "breakeven_price": breakeven, "prob_beats_now": prob})
    return pd.DataFrame(rows)


@dataclass
class Advice:
    recommendation: str
    months: int
    expected_net_ugx: float
    net_sell_now_ugx: float
    gain_ugx: float
    confidence: float | None
    confidence_label: str
    best_store_months: int | None          # storage option most likely to pay off
    best_store_prob: float | None
    options: pd.DataFrame


def advise(fc: forecasting.Forecast, qty: float, costs: Costs) -> Advice:
    opts = timing_options(fc, qty, costs)
    store = opts[opts["months"] > 0]
    probs = store["prob_beats_now"]
    have_probs = probs.notna().all() and len(probs) > 0

    best_store_months = best_store_prob = None
    if have_probs:
        i = probs.idxmax()
        best_store_months, best_store_prob = int(store.at[i, "months"]), float(probs[i])

    choice = opts.iloc[0]
    if len(store):
        top = store.loc[store["net_ugx"].idxmax()]
        prob_ok = (have_probs and top["prob_beats_now"] >= cfg.STORE_MIN_PROBABILITY)
        if top["gain_vs_now_ugx"] > 0 and (prob_ok or not have_probs):
            choice = top

    if not have_probs:
        conf = None
    elif choice["months"] == 0:
        conf = 1 - float(probs.max())
    else:
        conf = float(choice["prob_beats_now"])

    return Advice(
        recommendation=choice["action"], months=int(choice["months"]),
        expected_net_ugx=float(choice["net_ugx"]),
        net_sell_now_ugx=float(opts.iloc[0]["net_ugx"]),
        gain_ugx=float(choice["gain_vs_now_ugx"]),
        confidence=conf, confidence_label=confidence_label(conf),
        best_store_months=best_store_months, best_store_prob=best_store_prob,
        options=opts,
    )


# -----------------------------------------------------------------------------
# Market comparison
# -----------------------------------------------------------------------------
def sudden_jump(previous_price: float | None, latest_price: float) -> bool:
    """
    True if the latest price moved more than OUTLIER_MIN_JUMP (about 50%) in
    log terms from the previous recorded month. Uses only past data, so it can
    be applied at decision time. Phase 8 found the advice sending farmers up
    to 500 km to chase a one-month data error (Gulu millet, January 2024).
    """
    if previous_price is None or not np.isfinite(previous_price) or previous_price <= 0:
        return False
    return abs(np.log(latest_price / previous_price)) > cfg.OUTLIER_MIN_JUMP


def previous_observed_price(series: pd.DataFrame) -> float | None:
    """The recorded price before the latest one, or None."""
    obs = series["price_observed"].dropna()
    return float(obs.iloc[-2]) if len(obs) >= 2 else None


@dataclass
class MarketComparison:
    table: pd.DataFrame            # one row per comparable market, best first
    reference_month: pd.Timestamp
    excluded: list[str]            # markets left out because their latest price is from another month


def compare_markets(panel: pd.DataFrame, markets: pd.DataFrame, crop: str, home: str,
                    qty: float, horizon: int, costs: Costs,
                    registry: dict | None = None) -> MarketComparison:
    """
    Best option at each market whose latest price was recorded in the same
    month as the home market's, after transport from the home market.
    """
    registry = registry or forecasting.load_registry()
    crop_panel = panel[(panel["commodity"] == crop) & panel["price_observed"].notna()]
    latest = crop_panel.groupby("market")["date"].max()
    if home not in latest.index:
        raise ValueError(f"No {crop} prices at {home}.")
    ref = latest[home]
    gap = ((latest.dt.year - ref.year) * 12 + (latest.dt.month - ref.month)).abs()
    comparable = gap[gap <= cfg.COMPARE_MAX_MONTH_GAP].index.tolist()
    excluded = sorted(set(latest.index) - set(comparable))

    dist = road_km(markets, home)
    rows = []
    for m in comparable:
        series_m = pipeline.get_series(panel, crop, m)
        fc = forecasting.forecast_series(series_m, crop, m, horizon, registry)
        km = 0.0 if m == home else float(dist.get(m, np.nan))
        transport = costs.transport_ugx_per_kg_km * km * qty
        opts = timing_options(fc, qty, costs, transport_ugx=transport)
        best = opts.loc[opts["net_ugx"].idxmax()]
        rows.append({
            "market": m, "region": markets.set_index("market").at[m, "region"],
            "is_home": m == home, "road_km": km,
            "latest_price": fc.origin_price, "transport_ugx": transport,
            "best_option": best["action"], "best_months": int(best["months"]),
            "best_net_ugx": float(best["net_ugx"]),
            "sell_now_net_ugx": float(opts.iloc[0]["net_ugx"]),
            "sudden_jump": sudden_jump(previous_observed_price(series_m), fc.origin_price),
        })
    table = pd.DataFrame(rows)
    home_net = float(table.loc[table["is_home"], "sell_now_net_ugx"].iloc[0])
    table["vs_home_sell_now_ugx"] = table["best_net_ugx"] - home_net
    table = table.sort_values("best_net_ugx", ascending=False).reset_index(drop=True)
    return MarketComparison(table=table, reference_month=ref, excluded=excluded)


def best_alternative(mc: MarketComparison) -> pd.Series | None:
    """
    The best market other than home, or None. Markets whose latest price
    jumped suddenly are skipped: the jump may be a recording error, so the
    app does not send a farmer there on the strength of it.
    """
    alt = mc.table[~mc.table["is_home"] & ~mc.table["sudden_jump"]]
    return alt.iloc[0] if len(alt) else None
