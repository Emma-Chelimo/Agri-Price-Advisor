"""
Walk-forward backtest of every forecaster on every retail series.

Run from the project root (after scripts/build_dataset.py):

    python scripts/run_backtest.py            # baselines + current XGBoost
    python scripts/run_backtest.py --fast     # baselines only (about 20 s)
    python scripts/run_backtest.py --test     # also score the test split

By default only the validation split is scored. The test split is held back
until model choices are final (Phase 3), so it stays an honest final check.

Writes to processed/backtest/:
    forecasts.parquet       every forecast with its actual value
    metrics_by_series.csv   MAE, RMSE, MAPE, skill, direction per series
    summary.csv             one row per split, model and horizon
    summary_by_crop.csv     the same, broken down by crop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src import data, evaluation as ev, forecasters as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="baselines only")
    ap.add_argument("--test", action="store_true", help="also score the test split")
    args = ap.parse_args()

    panel = data.load_monthly()
    models = F.default_forecasters(include_xgb=not args.fast)
    splits = ("validation", "test") if args.test else ("validation",)

    print(f"Models: {', '.join(models)}")
    print(f"Splits: {', '.join(splits)}  (validation origins {cfg.VALIDATION_START} "
          f"to {cfg.TEST_START}, test origins from {cfg.TEST_START})")
    t0 = time.time()
    res = ev.run_and_save(panel, models, splits=splits)
    print(f"Done in {time.time() - t0:.0f}s\n")

    pd.set_option("display.width", 160)
    cols = ["split", "model", "horizon", "n_series", "n_pairs", "MAPE",
            "pooled_skill", "median_skill", "pct_beat", "direction_acc"]
    print("Summary (skill > 0 means better than 'same as last month')")
    print(res["summary"][cols].to_string(index=False))
    print("\nPooled skill vs naive, by horizon")
    print(ev.wide_skill(res["summary"]).to_string())
    print(f"\nFiles written to {cfg.BACKTEST_DIR}")


if __name__ == "__main__":
    main()
