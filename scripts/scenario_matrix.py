"""
Scenario matrix: every outcome the advice can give, and when it says "store".

    python scripts/scenario_matrix.py

Writes processed/scenarios/:
    current_advice.csv        every commodity and market on the latest prices (1,000 kg, default costs)
    store_thresholds.csv      forecast rise maize and beans need before storing is advised
    replay_by_month.csv       past replay: share of decisions advising storage, by commodity and calendar month
    replay_outcomes.csv       past replay: count of each outcome, by commodity
    cost_sensitivity.csv      share of markets advised to store as storage cost and losses change
    quantity_check.csv        the same advice for 100 kg and 10,000 kg
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import advice_service as S, config as cfg, decision, forecasting  # noqa: E402

OUT = cfg.PROCESSED_DIR / "scenarios"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
ctx = S.load_context()
QTY = 1000


def outcome(r) -> str:
    """The verdict as the farmer sees it."""
    if r.go_to_alternative:
        alt_months = int(r.alt["best_months"])
        how = "sell now" if alt_months == 0 else f"store {alt_months} month{'s' if alt_months > 1 else ''}"
        return f"Go to another market and {how}"
    if r.adv.months > 0:
        return f"Store {r.adv.months} month{'s' if r.adv.months > 1 else ''} at home"
    if r.adv.best_store_months and r.adv.best_store_prob is not None and r.adv.best_store_prob >= 0.4:
        return "Sell now at home (storing close but under 60%)"
    return "Sell now at home"


# ---------------------------------------------------------------- A. current advice
rows = []
pairs = ctx.summary[["commodity", "market"]].drop_duplicates().values.tolist()
for crop, market in pairs:
    try:
        r = S.get_advice(crop=crop, market=market, quantity_kg=QTY)
    except Exception as e:  # no history, etc.
        rows.append({"commodity": crop, "market": market, "outcome": f"No advice: {e}"})
        continue
    a = r.adv
    rows.append({
        "commodity": crop, "market": market, "model": r.fc.model,
        "latest_price": round(r.fc.origin_price), "price_month": str(r.fc.origin_date.date())[:7],
        "months_old": r.months_old, "stale": r.is_stale,
        "forecast_1_2_3": "/".join(str(round(v)) for v in r.fc.median),
        "outcome": outcome(r), "home_recommendation": a.recommendation,
        "best_store_months": a.best_store_months,
        "best_store_prob_pct": None if a.best_store_prob is None else round(100 * a.best_store_prob),
        "gain_ugx": round(a.gain_ugx), "confidence_shown_pct": None if a.confidence is None else round(100 * a.confidence),
        "confidence_label": a.confidence_label,
        "sell_market": r.sell_market,
        "alt_extra_ugx": None if r.alt is None else round(float(r.alt["best_net_ugx"]) - a.expected_net_ugx),
    })
cur = pd.DataFrame(rows)
cur.to_csv(OUT / "current_advice.csv", index=False)

# ---------------------------------------------------------------- B. thresholds
z = NormalDist().inv_cdf(cfg.STORE_MIN_PROBABILITY)   # 0.253
reg = ctx.registry["crops"]
th = []
for crop in ("Maize", "Beans"):
    p0 = float(cur.loc[cur["commodity"] == crop, "latest_price"].median())
    c = decision.Costs.default_for(crop)
    for h in cfg.HORIZONS:
        e = reg[crop]["horizons"][str(h)]
        sig, bias = e["log_sigma"], e["log_bias"]
        surv = (1 - c.loss_rate_per_month) ** h
        b = (p0 + c.storage_ugx_per_kg_month * h) / surv          # break-even price per kg sold
        need_median = np.log(b / p0)                               # forecast must at least cover costs
        need_60 = need_median + z * sig - bias                     # and give a 60% chance
        th.append({"commodity": crop, "months_stored": h, "typical_price": round(p0),
                   "storage_ugx_kg_month": c.storage_ugx_per_kg_month, "loss_pct_month": 100 * c.loss_rate_per_month,
                   "breakeven_price": round(b), "rise_to_cover_costs_pct": round(100 * (np.exp(need_median) - 1), 1),
                   "rise_needed_for_store_advice_pct": round(100 * (np.exp(need_60) - 1), 1),
                   "model_sigma": sig, "model_bias": bias})
thr = pd.DataFrame(th)
thr.to_csv(OUT / "store_thresholds.csv", index=False)

# ---------------------------------------------------------------- C. past replay
d = pd.read_parquet(cfg.ADVICE_EVAL_DIR / "decisions.parquet")
d["stores"] = d["advice_months"] > 0
by_month = (d.groupby(["crop", "calendar_month"])["stores"].mean().mul(100).round(1)
            .unstack("calendar_month"))
by_month.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in by_month.columns]
by_month.to_csv(OUT / "replay_by_month.csv")
d["kind"] = np.select([d["moved_market"], d["stores"]], ["Another market", "Store at home"], "Sell now at home")
outc = (d.groupby(["crop", "kind"]).size().unstack(fill_value=0))
outc["decisions"] = outc.sum(axis=1)
store_k = d[d["stores"]].groupby(["crop", "advice_months"]).size().unstack(fill_value=0)
store_k.columns = [f"store_{c}m" for c in store_k.columns]
outc = outc.join(store_k).fillna(0).astype(int)
outc.to_csv(OUT / "replay_outcomes.csv")
right = d[d["stores"] & d["advice_right"].notna()].groupby("crop")["advice_right"].agg(["size", "mean"])

# ---------------------------------------------------------------- D. cost sensitivity (home timing only)
sens = []
for crop in ("Maize", "Beans"):
    mk = cur[(cur["commodity"] == crop) & cur["model"].notna()]["market"].tolist()
    fcs = {m: forecasting.forecast(ctx.panel, crop, m, cfg.MAX_FORECAST_HORIZON, ctx.registry) for m in mk}
    base = decision.Costs.default_for(crop)
    for storage in (0, 5, 10, 20, 30, 40):
        for lmult in (0, 0.5, 1, 2):
            c = decision.Costs(storage, base.loss_rate_per_month * lmult)
            advs = [decision.advise(f, QTY, c) for f in fcs.values()]
            sens.append({"commodity": crop, "storage_ugx_kg_month": storage,
                         "loss_pct_month": round(100 * c.loss_rate_per_month, 3),
                         "markets": len(advs), "advised_store": sum(a.months > 0 for a in advs),
                         "store_pct": round(100 * np.mean([a.months > 0 for a in advs]), 1)})
sen = pd.DataFrame(sens)
sen.to_csv(OUT / "cost_sensitivity.csv", index=False)

# ---------------------------------------------------------------- E. quantity does not change the verdict
qc = []
for crop, market in [("Maize", "Gulu"), ("Beans", "Lira"), ("Sorghum", "Gulu")]:
    for q in (100, 1000, 10000):
        r = S.get_advice(crop=crop, market=market, quantity_kg=q)
        qc.append({"commodity": crop, "market": market, "kg": q, "outcome": outcome(r),
                   "gain_per_kg": round(r.adv.gain_ugx / q, 2)})
qcd = pd.DataFrame(qc)
qcd.to_csv(OUT / "quantity_check.csv", index=False)

# ---------------------------------------------------------------- print
print("A. CURRENT ADVICE (1,000 kg, default costs): outcome counts by commodity")
print(cur.groupby(["commodity", "outcome"]).size().unstack(fill_value=0).to_string())
print("\nMaize and beans detail")
print(cur[cur["commodity"].isin(["Maize", "Beans"])][["commodity", "market", "price_month", "stale", "forecast_1_2_3",
      "latest_price", "outcome", "best_store_months", "best_store_prob_pct", "confidence_label", "sell_market"]].to_string(index=False))
print("\nB. STORE THRESHOLDS")
print(thr.to_string(index=False))
print("\nC. PAST REPLAY: outcomes by commodity")
print(outc.to_string())
print("\nStore advice that turned out right (pre-test and test together)")
print(right.round(3).to_string())
print("\nShare of past decisions advising storage, by calendar month (%)")
print(by_month.to_string())
print("\nD. COST SENSITIVITY (share of current markets advised to store at home)")
print(sen.pivot_table(index=["commodity", "storage_ugx_kg_month"], columns="loss_pct_month", values="store_pct").to_string())
print("\nE. QUANTITY CHECK")
print(qcd.to_string(index=False))
