"""
Evaluate the advice on past prices (Phase 8, Objective 8).

For every past month t at every market with a recorded price, pretend a
farmer holds EVAL_QUANTITY_KG kg there and asks the app for advice. The
system sees only prices up to t (each series is cut at t before it is
forecast). Then the advice is scored with the prices that actually came
after, and compared with simpler strategies:

    harvest_sale    sell everything at t at the farmer's market (the baseline
                    Objective 8 asks for)
    timing_advice   follow the sell-now-or-store advice at the farmer's market
    full_advice     follow the timing advice, and go to the best alternative
                    market whenever the dashboard shows it pays more after
                    transport. A move is scored at that market's price
                    EVAL_ALT_MARKET_LAG_MONTHS later, because the farmer only
                    learns the price from published data and then has to
                    travel, so the price may have changed by then.
    store_3, store_6  always store 3 or 6 months, then sell at home
    hindsight       the best timing in hindsight (upper bound, not achievable)

Earnings use the same formulas and default costs as the app: storage cost
and losses per crop, transport per kg per km for other markets. The price on
a sale month is the observed WFP price, or the interpolated value for gaps of
up to two months. If no price exists for the sale month, that strategy is not
scored at that decision point (counts are reported).

The confidence check: for each decision, the app states a confidence (the
chance its timing choice is the better one). We record whether, in reality,
it was: for "sell now", no storage option with a known outcome beat selling
now; for "store k", storing k months beat selling now. Grouping decisions by
stated confidence shows whether the percentages mean what they say.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from . import decision, forecasting

STRATEGIES = ["harvest_sale", "timing_advice", "full_advice", "store_3", "store_6", "hindsight"]


@dataclass
class _Series:
    df: pd.DataFrame                 # rows sorted by date, positional index
    pos: dict                        # date -> row position
    price: np.ndarray                # observed price, or interpolated, NaN in long gaps
    observed: np.ndarray             # True where WFP recorded a price
    n_obs_cum: np.ndarray


def _prep(panel: pd.DataFrame) -> dict:
    out = {}
    for key, g in panel.groupby(["commodity", "market"], sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        obs = g["price_observed"].notna().to_numpy()
        out[key] = _Series(df=g, pos={d: i for i, d in enumerate(g["date"])},
                           price=g["price"].to_numpy(dtype=float), observed=obs,
                           n_obs_cum=np.cumsum(obs))
    return out


def _price_at(s: _Series, date) -> float | None:
    i = s.pos.get(date)
    if i is None:
        return None
    p = s.price[i]
    return float(p) if np.isfinite(p) else None


def evaluate(panel: pd.DataFrame, markets: pd.DataFrame, registry: dict | None = None,
             start: str | None = None, end: str | None = None,
             crops: list[str] | None = None, verbose: bool = True,
             transport_rate: float | None = None,
             alt_lag_months: int | None = None,
             check_jumps: bool = True) -> pd.DataFrame:
    """One row per decision point (crop, market, month) with every strategy's earnings."""
    rate = cfg.TRANSPORT_UGX_PER_KG_KM if transport_rate is None else transport_rate
    lag = cfg.EVAL_ALT_MARKET_LAG_MONTHS if alt_lag_months is None else alt_lag_months
    registry = registry or forecasting.load_registry()
    q = cfg.EVAL_QUANTITY_KG
    H = cfg.MAX_FORECAST_HORIZON
    series = _prep(panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE])
    dist = {m: decision.road_km(markets, m) for m in markets["market"]}
    region = markets.set_index("market")["region"].to_dict()
    start_ts = pd.Period(start, "M").to_timestamp() if start else None
    end_ts = pd.Period(end, "M").to_timestamp() if end else None

    rows = []
    t0 = time.time()
    for crop in (crops or cfg.TARGET_CROPS):
        costs = decision.Costs.default_for(crop)
        surv = (1 - costs.loss_rate_per_month) ** np.arange(H + 1)
        storage = costs.storage_ugx_per_kg_month * q * np.arange(H + 1)
        crop_series = {m: s for (c, m), s in series.items() if c == crop}
        if not crop_series:
            continue
        # every month with at least one observed price for this crop
        months = sorted({d for s in crop_series.values()
                         for d, o in zip(s.df["date"], s.observed) if o})
        harvest = set(cfg.HARVEST_MONTHS.get(crop, []))

        for t in months:
            if (start_ts and t < start_ts) or (end_ts and t > end_ts):
                continue
            # markets with a recorded price this month (the app's same-month rule)
            here = {m: s for m, s in crop_series.items()
                    if (i := s.pos.get(t)) is not None and s.observed[i]}
            # forecast and best option at each of these markets, from history up to t
            best = {}
            for m, s in here.items():
                i = s.pos[t]
                if s.n_obs_cum[i] < cfg.EVAL_MIN_HISTORY_MONTHS:
                    continue
                fc = forecasting.forecast_series(s.df.iloc[: i + 1], crop, m, H, registry)
                adv = decision.advise(fc, q, costs)
                # best option by median earnings, before any transport
                net = adv.options["net_ugx"].to_numpy()
                k_best = int(np.argmax(net))
                prev_i = np.flatnonzero(s.observed[:i])
                prev = float(s.df.at[prev_i[-1], "price_observed"]) if len(prev_i) else None
                jump = check_jumps and decision.sudden_jump(prev, fc.origin_price)
                best[m] = (fc, adv, k_best, float(net[k_best]), jump)

            for home, (fc, adv, _, _, _) in best.items():
                p0 = fc.origin_price
                baseline = p0 * q

                def realised(m: str, k: int, transport: float = 0.0, delay: int = 0) -> float:
                    """Store k months, then sell at m `delay` months later."""
                    p = _price_at(crop_series[m], t + pd.DateOffset(months=k + delay))
                    if p is None:
                        return np.nan
                    return p * q * surv[k] - storage[k] - transport

                k_adv = adv.months
                timing = realised(home, k_adv)

                # best alternative market, as the dashboard would show it
                alt_m, alt_k, alt_net, alt_tr = None, None, -np.inf, 0.0
                for m, (_, _, k_m, net_m, jump_m) in best.items():
                    if m == home or jump_m:
                        continue
                    tr = rate * float(dist[home].get(m, np.nan)) * q
                    if np.isfinite(tr) and net_m - tr > alt_net:
                        alt_m, alt_k, alt_net, alt_tr = m, k_m, net_m - tr, tr
                go_alt = alt_m is not None and alt_net > adv.expected_net_ugx
                full = realised(alt_m, alt_k, alt_tr, delay=lag) if go_alt else timing

                store = {k: realised(home, k) for k in range(H + 1)}
                known = {k: v for k, v in store.items() if np.isfinite(v)}
                hindsight = max(known.values()) if known else np.nan

                # was the stated confidence borne out?
                right = np.nan
                if adv.confidence is not None:
                    if k_adv == 0:
                        later = [v for k, v in known.items() if k > 0]
                        if later:
                            right = float(all(v <= store[0] for v in later))
                    elif np.isfinite(store.get(k_adv, np.nan)):
                        right = float(store[k_adv] > store[0])

                rows.append({
                    "crop": crop, "market": home, "region": region.get(home),
                    "decision_month": t, "calendar_month": t.month,
                    "is_harvest_month": t.month in harvest,
                    "period": "test" if t >= pd.Period(cfg.TEST_START, "M").to_timestamp() else "pre-test",
                    "price_now": p0,
                    "advice": adv.recommendation, "advice_months": k_adv,
                    "advice_market": alt_m if go_alt else home, "moved_market": go_alt,
                    "alt_road_km": float(dist[home].get(alt_m, np.nan)) if go_alt else 0.0,
                    "confidence": adv.confidence, "advice_right": right,
                    "harvest_sale": baseline, "timing_advice": timing, "full_advice": full,
                    "store_3": store.get(3, np.nan), "store_6": store.get(6, np.nan),
                    "hindsight": hindsight,
                })
        if verbose:
            print(f"  {crop}: {sum(r['crop'] == crop for r in rows):,} decision points, "
                  f"{time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def summarise(ev: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """
    Per group: average earnings per 1,000 kg, average gain over the harvest
    sale (UGX and %), and how often each strategy beat, matched or lost to it.
    Each strategy is scored only where its outcome is known; n says how often.
    """
    by = by or []
    rows = []
    groups = ev.groupby(by) if by else [((), ev)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        base = g["harvest_sale"]
        row = dict(zip(by, key))
        row["decisions"] = len(g)
        row["advised_store_pct"] = round(100 * (g["advice_months"] > 0).mean(), 1)
        row["advised_other_market_pct"] = round(100 * g["moved_market"].mean(), 1)
        row["avg_harvest_sale_ugx"] = round(base.mean())
        for s in STRATEGIES[1:]:
            ok = g[s].notna()
            gain = (g.loc[ok, s] - base[ok])
            row[f"{s}_n"] = int(ok.sum())
            row[f"{s}_gain_ugx"] = round(gain.mean()) if ok.any() else np.nan
            row[f"{s}_gain_pct"] = (round(100 * gain.sum() / base[ok].sum(), 2)
                                    if ok.any() else np.nan)
            row[f"{s}_better_pct"] = round(100 * (gain > 1).mean(), 1) if ok.any() else np.nan
            row[f"{s}_worse_pct"] = round(100 * (gain < -1).mean(), 1) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def worst_cases(ev: pd.DataFrame, n: int = 10, strategy: str = "full_advice") -> pd.DataFrame:
    g = ev[ev[strategy].notna()].copy()
    g["gain_ugx"] = g[strategy] - g["harvest_sale"]
    g["gain_pct"] = 100 * g["gain_ugx"] / g["harvest_sale"]
    cols = ["crop", "market", "decision_month", "advice", "advice_market", "alt_road_km",
            "price_now", "harvest_sale", strategy, "gain_ugx", "gain_pct"]
    out = g.nsmallest(n, "gain_ugx")[cols]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(1)
    return out


def calibration(ev: pd.DataFrame, bins=(0.0, 0.55, 0.6, 0.75, 1.0001)) -> pd.DataFrame:
    """Stated confidence against how often the timing advice was actually right."""
    g = ev[ev["confidence"].notna() & ev["advice_right"].notna()].copy()
    g["band"] = pd.cut(g["confidence"], bins=list(bins), right=False)
    out = g.groupby("band", observed=True).agg(
        decisions=("advice_right", "size"),
        stated_confidence_pct=("confidence", lambda x: round(100 * x.mean(), 1)),
        actually_right_pct=("advice_right", lambda x: round(100 * x.mean(), 1)),
    ).reset_index()
    out["band"] = out["band"].astype(str)
    return out


def run_and_save(panel, markets, registry=None, out_dir=None, **kw) -> dict:
    out_dir = out_dir or cfg.ADVICE_EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = evaluate(panel, markets, registry, **kw)
    harvest = ev[ev["is_harvest_month"]]
    res = {
        "decisions": ev,
        "summary_harvest": summarise(harvest, ["crop"]),
        "summary_harvest_all_crops": summarise(harvest),
        "summary_all_months": summarise(ev, ["crop"]),
        "summary_by_period": summarise(harvest, ["period"]),
        "worst_cases": worst_cases(harvest),
        "calibration": calibration(ev),
    }
    ev.to_parquet(out_dir / "decisions.parquet", index=False)
    for k, v in res.items():
        if k != "decisions":
            v.to_csv(out_dir / f"{k}.csv", index=False)
    return res
