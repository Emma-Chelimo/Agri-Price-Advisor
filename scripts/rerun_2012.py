"""
Rerun the full comparison (current system vs SARIMA hybrid) on data from
January 2012 to December 2025, forecasting 1 to 3 months ahead.

    python scripts/rerun_2012.py
    RERUN_START=2008-01-01 RERUN_OUT=rerun_2008_h3 python scripts/rerun_2012.py   # all data from 2008

Splits (forecast origins):
    warm-up      2012-2014   history only (each series needs 24 real months)
    calibration  2015-2021   SARIMA order choice, shrinkage lambda, residual-model history
    validation   2022-2023   choose one model per crop
    test         2024-2025   scored once

Everything is written to processed/rerun_2012/; the app's own files are not touched.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KERAS_BACKEND", "jax")

from src import config as cfg  # noqa: E402

# ---- settings for this rerun (before the model modules are imported) -----------
DATA_START = os.environ.get("RERUN_START", "2012-01-01")
DATA_END = os.environ.get("RERUN_END", "2025-12-01")
OUT = cfg.PROCESSED_DIR / os.environ.get("RERUN_OUT", "rerun_2012")
OUT.mkdir(parents=True, exist_ok=True)
cfg.HORIZONS = [1, 2, 3]
cfg.MAX_FORECAST_HORIZON = 3
cfg.CALIBRATION_START = "2015-01"
cfg.BACKTEST_DIR = OUT
cfg.POOLED_LAMBDAS_JSON = OUT / "pooled_lambdas.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data, evaluation as ev, forecasters as F, hybrid as HY, pooled as P  # noqa: E402

pd.set_option("display.width", 220)
KEYS = ["commodity", "market", "origin", "horizon"]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)


panel = data.load_monthly()
panel = panel[(panel["date"] >= DATA_START) & (panel["date"] <= DATA_END)].copy()
r = panel[panel["pricetype"] == cfg.MODEL_PRICE_TYPE]
log(f"Data {DATA_START[:7]} to {DATA_END[:7]}: {r['price_observed'].notna().sum()} real retail prices, "
    f"{r.groupby(['commodity', 'market']).ngroups} series")

# ---- 1. current system ----------------------------------------------------------
base = P.PooledGBM(panel)
cal = ev.backtest(panel, {"naive": F.naive, "pooled_gbm": base}, split="calibration", verbose=False)
lambdas = P.fit_lambdas(cal)
P.save_lambdas(lambdas)
log(f"Calibration pairs {len(cal) // 2}; lambda by horizon {lambdas[next(iter(lambdas))]}")
current = {"naive": F.naive, "seasonal_naive": F.seasonal_naive, "app_baseline": F.app_baseline,
           "pooled_gbm": base, "pooled_shrunk": P.PooledShrunk(base, lambdas)}
main = pd.concat([ev.backtest(panel, current, split=s, verbose=False)
                  for s in ("validation", "test")], ignore_index=True)
log(f"Current-system backtest done: {main[main.model == 'naive'].groupby('split').size().to_dict()}")

# ---- 2. SARIMA order chosen on calibration ----------------------------------------
tables, cal_skill = {}, {}
nv = cal[cal["model"] == "naive"][KEYS + ["actual", "origin_price"]]
for o in ("A", "B", "C"):
    s = HY.sarima_table(panel, order=o)
    tables[o] = s
    long = s.melt(id_vars=["commodity", "market", "origin"], value_vars=[f"sar{h}" for h in cfg.HORIZONS],
                  var_name="hh", value_name="lc")
    long["horizon"] = long["hh"].str[3:].astype(int)
    m = nv.merge(long[KEYS + ["lc"]], on=KEYS)
    fc = m["origin_price"] * np.exp(m["lc"])
    cal_skill[o] = 1 - (fc - m["actual"]).abs().sum() / (m["origin_price"] - m["actual"]).abs().sum()
    log(f"SARIMA order {o} {HY.SARIMA_ORDERS[o]}: calibration skill {cal_skill[o]:+.4f}")
best = max(cal_skill, key=cal_skill.get)
log(f"Chosen SARIMA order: {best}")
sar = tables[best]
sar.to_parquet(OUT / "sarima_table.parquet", index=False)

# ---- 3. residual models (LSTM + XGBoost), quarterly from 2015 -------------------
t = HY.attach_residuals(sar, panel)
feats = P.build_features(panel)
fit_dates = list(pd.date_range(cfg.CALIBRATION_START + "-01", t["origin"].max(), freq="QS"))
t = HY.run_residual_models(t, feats, fit_dates, verbose=False)
t.to_parquet(OUT / "residual_predictions.parquet", index=False)
hyb = HY.to_forecasts(t)
log("Residual models done")

# ---- 4. same pairs, scoring, selection -----------------------------------------
def on_pairs(b):
    pairs = b[b["model"] == "naive"].drop(columns=["model", "forecast"])
    h = pairs.merge(hyb, on=KEYS, how="inner")
    return ev.common_pairs(pd.concat([b, h[b.columns]], ignore_index=True))


both = on_pairs(main)
calib = on_pairs(cal)
by_crop = ev.summary(both, by=["commodity"])
val = by_crop[by_crop["split"] == "validation"].groupby(["commodity", "model"])["pooled_skill"].mean()
MARGIN = 0.02


def choose(pool):
    pick = {}
    for crop in sorted(both["commodity"].unique()):
        s = val.loc[crop]
        s = s[s.index.isin(pool)]
        b = s.idxmax()
        pick[crop] = b if (s[b] > MARGIN and b != "naive") else "naive"
    return pick


pools = {
    "current_system": ["naive", "seasonal_naive", "pooled_gbm", "pooled_shrunk"],
    "hybrid_system": ["naive", "sarima", "sarima_lstm", "sarima_xgb", "hybrid"],
    "everything": ["naive", "seasonal_naive", "pooled_gbm", "pooled_shrunk",
                   "sarima", "sarima_lstm", "sarima_xgb", "hybrid"],
}
picks = {k: choose(v) for k, v in pools.items()}
extra = [ev.selected_forecasts(both, {"crops": {c: {"model": m} for c, m in p.items()}})
         .query("model == 'selected'").assign(model=name) for name, p in picks.items()]
allfc = pd.concat([both] + extra, ignore_index=True)

summ = ev.summary(allfc)
dm = ev.decision_metrics(allfc)
crop = ev.summary(allfc, by=["commodity"])
cs = ev.summary(calib)
allfc.to_parquet(OUT / "forecasts.parquet", index=False)
summ.to_csv(OUT / "summary.csv", index=False)
dm.to_csv(OUT / "decision_metrics.csv", index=False)
crop.to_csv(OUT / "summary_by_crop.csv", index=False)
cs.to_csv(OUT / "calibration_summary.csv", index=False)
(OUT / "picks.json").write_text(json.dumps({"sarima_order": best, "calibration_skill": cal_skill,
                                            **picks}, indent=2))

order = ["naive", "seasonal_naive", "app_baseline", "sarima", "sarima_lstm", "sarima_xgb", "hybrid",
         "pooled_gbm", "pooled_shrunk", "current_system", "hybrid_system", "everything"]


def wide(s, value):
    w = s.pivot_table(index=["split", "model"], columns="horizon", values=value)
    w["mean"] = w.mean(axis=1)
    w = w.reset_index()
    w["o"] = w["model"].map({m: i for i, m in enumerate(order)})
    return w.sort_values(["split", "o"]).drop(columns="o").round(3)


print("\nPairs per split:", allfc[allfc.model == "naive"].groupby("split").size().to_dict(),
      "calibration:", len(calib[calib.model == "naive"]))
print("Series per split:", allfc[allfc.model == "naive"].assign(s=lambda d: d.commodity + d.market)
      .groupby("split")["s"].nunique().to_dict())
for title, col in (("SKILL vs naive", "pooled_skill"), ("MAPE %", "MAPE"),
                   ("Direction accuracy", "direction_acc"), ("Share of series beating naive %", "pct_beat")):
    print(f"\n{title}")
    print(wide(summ, col).to_string(index=False))
print("\nCalibration (2015-2021) skill")
print(wide(cs, "pooled_skill").to_string(index=False))
print("\nValidation mean skill by crop")
vu = val.unstack().round(3)
print(vu[[m for m in order if m in vu.columns]].to_string())
print("\nPicks:", json.dumps(picks, indent=1))
tc = crop[crop["split"] == "test"].groupby(["commodity", "model"])["pooled_skill"].mean().unstack().round(3)
print("\nTest mean skill by crop")
print(tc[[m for m in order if m in tc.columns]].to_string())
d = dm[dm["model"].isin(["current_system", "hybrid_system", "everything", "hybrid", "sarima"])]
print("\nDecision metrics")
print(d[["split", "model", "horizon", "band_coverage", "confident_share", "confident_dir_acc",
         "up_call_hit"]].to_string(index=False))
log("Done")
