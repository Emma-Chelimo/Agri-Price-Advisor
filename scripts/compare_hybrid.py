"""
Compare the SARIMA hybrid with the current system on identical pairs.

    python scripts/run_hybrid.py      # first
    python scripts/compare_hybrid.py

Writes processed/hybrid/comparison_*.csv and prints the tables.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from src import config as cfg, evaluation as ev

pd.set_option("display.width", 220)
out = cfg.PROCESSED_DIR / "hybrid"
hyb = pd.read_parquet(out / "hybrid_forecasts_raw.parquet")
main = pd.read_parquet(cfg.BACKTEST_DIR / "forecasts.parquet")
cal = pd.read_parquet(cfg.BACKTEST_DIR / "calibration_forecasts.parquet")
registry = json.loads(cfg.MODEL_REGISTRY_JSON.read_text()) if hasattr(cfg, "MODEL_REGISTRY_JSON") \
    else json.loads((cfg.PROCESSED_DIR / "model_registry.json").read_text())

KEYS = ["commodity", "market", "origin", "horizon"]


def on_pairs(base: pd.DataFrame) -> pd.DataFrame:
    pairs = base[base["model"] == "naive"].drop(columns=["model", "forecast"])
    h = pairs.merge(hyb, on=KEYS, how="inner")
    return ev.common_pairs(pd.concat([base, h[base.columns]], ignore_index=True))


both = on_pairs(main)
both = pd.concat([both, ev.selected_forecasts(both, registry).query("model == 'selected'")],
                 ignore_index=True)
calib = on_pairs(cal)

# ---- selection on validation, same rule as scripts/select_models.py -------------
MARGIN = 0.02
by_crop = ev.summary(both[both["model"] != "selected"], by=["commodity"])
val = by_crop[by_crop["split"] == "validation"].groupby(["commodity", "model"])["pooled_skill"].mean()


def choose(pool):
    pick = {}
    for crop in sorted(both["commodity"].unique()):
        s = val.loc[crop]
        s = s[s.index.isin(pool)]
        best = s.idxmax()
        pick[crop] = best if s[best] > MARGIN and best != "naive" else "naive"
    return pick


hybrid_family = ["naive", "sarima", "sarima_lstm", "sarima_xgb", "hybrid"]
everything = ["naive", "seasonal_naive", "app_baseline", "xgb_current", "pooled_gbm",
              "pooled_shrunk", "sarima", "sarima_lstm", "sarima_xgb", "hybrid"]
picks = {"selected_hybrid_only": choose(hybrid_family), "selected_all": choose(everything)}
extra = []
for name, pick in picks.items():
    reg = {"crops": {c: {"model": m} for c, m in pick.items()}}
    extra.append(ev.selected_forecasts(both, reg).query("model == 'selected'").assign(model=name))
allfc = pd.concat([both] + extra, ignore_index=True)

summ = ev.summary(allfc)
dm = ev.decision_metrics(allfc)
crop = ev.summary(allfc, by=["commodity"])
summ.to_csv(out / "comparison_summary.csv", index=False)
dm.to_csv(out / "comparison_decision.csv", index=False)
crop.to_csv(out / "comparison_by_crop.csv", index=False)
cs = ev.summary(calib)
cs.to_csv(out / "comparison_calibration.csv", index=False)

order = ["naive", "seasonal_naive", "sarima", "sarima_lstm", "sarima_xgb", "hybrid",
         "pooled_gbm", "pooled_shrunk", "xgb_current", "app_baseline",
         "selected", "selected_hybrid_only", "selected_all"]


def wide(s, value):
    w = s.pivot_table(index=["split", "model"], columns="horizon", values=value)
    w["mean"] = w.mean(axis=1)
    w = w.reset_index()
    w["o"] = w["model"].map({m: i for i, m in enumerate(order)})
    return w.sort_values(["split", "o"]).drop(columns="o").round(3)


print("Pairs per split:", allfc[allfc.model == "naive"].groupby("split").size().to_dict())
print("\nSKILL vs naive (pooled, 1 - total abs error / naive's)")
print(wide(summ, "pooled_skill").to_string(index=False))
print("\nMAPE %")
print(wide(summ, "MAPE").to_string(index=False))
print("\nDirection accuracy")
print(wide(summ, "direction_acc").to_string(index=False))
print("\nCalibration period (2019-2021) skill")
print(wide(cs, "pooled_skill").to_string(index=False))
print("\nValidation mean skill by crop")
print(val.unstack().round(3)[[m for m in order if m in val.unstack().columns]].to_string())
print("\nPicks:", json.dumps(picks, indent=1))
tc = crop[crop["split"] == "test"].groupby(["commodity", "model"])["pooled_skill"].mean().unstack().round(3)
print("\nTest mean skill by crop")
print(tc[[m for m in order if m in tc.columns]].to_string())
dsel = dm[dm["model"].isin(["selected", "selected_hybrid_only", "selected_all", "hybrid", "sarima"])]
print("\nDecision metrics (test)")
print(dsel[dsel.split == "test"][["model", "horizon", "band_coverage", "confident_share",
                                   "confident_dir_acc", "up_call_hit"]].to_string(index=False))
(out / "picks.json").write_text(json.dumps(picks, indent=2))
