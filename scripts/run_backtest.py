"""
Walk-forward backtest of every forecaster on every retail series.

Run from the project root (after scripts/build_dataset.py):

    python scripts/run_backtest.py --test     # everything (about 15 minutes)
    python scripts/run_backtest.py            # validation split only
    python scripts/run_backtest.py --no-xgb   # skip the slow notebook XGBoost
    python scripts/run_backtest.py --fast     # baselines only (about 20 s),
                                              # saved to processed/backtest_fast/

Steps
-----
1. Calibration (origins 2015-2021): run the pooled model and naive, then fit
   the shrinkage towards naive for each crop and horizon. Saved to
   processed/pooled_lambdas.json and backtest/calibration_forecasts.parquet.
2. Validation (2022-2023) and, with --test, test (2024-2025): score every model,
   including the shrunk pooled model, on the same pairs.

Writes to processed/backtest/:
    forecasts.parquet           every forecast with its actual value
    metrics_by_series.csv       MAE, RMSE, MAPE, skill, direction per series
    summary.csv                 one row per split, model and horizon
    summary_by_crop.csv         the same, broken down by crop
    decision_metrics.csv        confident-call accuracy and band coverage
    headline.csv                average over horizons 1-3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data, evaluation as ev, forecasters as F, pooled as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="baselines only")
    ap.add_argument("--no-xgb", action="store_true", help="skip the notebook XGBoost")
    ap.add_argument("--test", action="store_true", help="also score the test split")
    args = ap.parse_args()

    panel = data.eval_window(data.load_monthly())   # 2008 to Dec 2025
    models = F.default_forecasters(include_xgb=not (args.fast or args.no_xgb))
    splits = ("validation", "test") if args.test else ("validation",)
    out_dir = cfg.PROCESSED_DIR / "backtest_fast" if args.fast else cfg.BACKTEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.fast:
        print("Building pooled features...")
        base = P.PooledGBM(panel)
        print(f"Calibration split (origins {cfg.CALIBRATION_START} to {cfg.VALIDATION_START})")
        cal = ev.backtest(panel, {"naive": F.naive, "pooled_gbm": base}, split="calibration")
        cal.to_parquet(out_dir / "calibration_forecasts.parquet", index=False)
        lambdas = P.fit_lambdas(cal)
        P.save_lambdas(lambdas, {c: int(n) for c, n in
                                 cal[cal["model"] == "pooled_gbm"].groupby("commodity").size().items()})
        print("Shrinkage lambda by crop and horizon (0 = naive, 1 = raw pooled model)")
        print(pd.DataFrame(lambdas).T.to_string(), "\n")
        models["pooled_gbm"] = base
        models["pooled_shrunk"] = P.PooledShrunk(base, lambdas)

    print(f"Models: {', '.join(models)}")
    print(f"Splits: {', '.join(splits)}  (validation origins {cfg.VALIDATION_START} "
          f"to {cfg.TEST_START}, test origins from {cfg.TEST_START})")
    fc = pd.concat([ev.backtest(panel, models, split=s) for s in splits], ignore_index=True)
    res = ev.save_results(fc, out_dir)
    print(f"Done in {time.time() - t0:.0f}s\n")

    pd.set_option("display.width", 200)
    cols = ["split", "model", "horizon", "n_series", "n_pairs", "MAPE",
            "pooled_skill", "median_skill", "pct_beat", "direction_acc"]
    print("Summary (skill > 0 means better than 'same as last month')")
    print(res["summary"][cols].to_string(index=False))
    print("\nAverage over 1-3 months ahead")
    print(res["headline"].to_string(index=False))
    print("\nDecision metrics")
    print(res["decision_metrics"].to_string(index=False))
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
