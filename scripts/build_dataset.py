"""
Rebuild every processed data file from the raw WFP CSV.

Run from the project root:

    python scripts/build_dataset.py

Writes to processed/:
    cleaned_prices.parquet   cleaned rows, retail and wholesale
    monthly_prices.parquet   retail panel on a complete monthly calendar
    series_summary.csv       coverage, gaps and staleness per series
    latest_prices.csv        latest observed price per series, with its age
    markets.csv              market coordinates and regions
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src.data import build_all  # noqa: E402


def main() -> None:
    out = build_all(write=True)
    cleaned, panel, summary = out["cleaned"], out["panel"], out["summary"]

    pd.set_option("display.width", 140)
    print("Cleaning steps")
    print(out["cleaning_log"].to_string(index=False))

    print(f"\nMarkets kept: {cleaned['market'].nunique()} "
          f"(refugee settlements excluded)")
    print("Crops:", ", ".join(sorted(cleaned["commodity"].unique())))
    print("Markets per region:",
          out["markets"].groupby("region").size().to_dict())

    print(f"\nRetail monthly panel: {len(panel):,} rows, "
          f"{len(summary)} series, reference month {summary.attrs['reference_month']}")
    print(f"  observed months:     {int(panel['price_observed'].notna().sum()):,}")
    print(f"  interpolated months: {int(panel['is_imputed'].sum()):,} "
          f"(gaps of {cfg.MAX_INTERPOLATE_GAP} months or less)")
    print(f"  long-gap months:     {int(panel['in_long_gap'].sum()):,} (left missing)")
    print(f"  outlier flags:       {int(panel['is_outlier_flag'].sum())} (review, not removed)")

    stale = summary[summary["is_stale"]]
    print(f"\nStale series (no price for more than {cfg.STALE_AFTER_MONTHS} months): "
          f"{len(stale)} of {len(summary)}")

    print("\nSeries per crop (fresh / stale):")
    by_crop = summary.groupby("commodity")["is_stale"].agg(
        fresh=lambda s: int((~s).sum()), stale="sum")
    print(by_crop.to_string())

    print(f"\nFiles written to {cfg.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
