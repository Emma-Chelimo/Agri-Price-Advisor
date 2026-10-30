"""
Objective 8: test the advice on past prices.

Run from the project root (after build_dataset, run_backtest and select_models):

    python scripts/evaluate_advice.py

Replays every past month at every market as if a farmer with 1,000 kg asked
for advice, using only data up to that month, then scores the advice with
the prices that followed. Writes to processed/advice_eval/:

    decisions.parquet               every decision point and strategy outcome
    summary_harvest.csv             harvest months, by crop
    summary_harvest_all_crops.csv   harvest months, all crops together
    summary_all_months.csv          every month, by crop
    summary_by_period.csv           harvest months, before vs during the test period
    worst_cases.csv                 the largest losses from following the advice
    calibration.csv                 stated confidence vs how often it was right
    sensitivity.csv                 main result under other assumptions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import advice_eval as AE  # noqa: E402
from src import config as cfg  # noqa: E402
from src import data, forecasting  # noqa: E402


def headline(ev: pd.DataFrame, label: str) -> dict:
    h = AE.summarise(ev[ev["is_harvest_month"]]).iloc[0]
    return {"scenario": label, "decisions": int(h["decisions"]),
            "advised_other_market_pct": h["advised_other_market_pct"],
            "full_advice_gain_pct": h["full_advice_gain_pct"],
            "full_advice_better_pct": h["full_advice_better_pct"],
            "full_advice_worse_pct": h["full_advice_worse_pct"]}


def main() -> None:
    panel, markets = data.load_monthly(), data.load_markets()
    registry = forecasting.load_registry()
    t0 = time.time()
    print("Main evaluation")
    res = AE.run_and_save(panel, markets, registry)
    ev = res["decisions"]

    print("Sensitivity checks")
    sens = [headline(ev, f"main (transport {cfg.TRANSPORT_UGX_PER_KG_KM:g} UGX/kg/km, "
                         f"{cfg.EVAL_ALT_MARKET_LAG_MONTHS}-month delay, jump guard on)")]
    for label, kw in [
        ("without the sudden-jump guard", dict(check_jumps=False)),
        ("transport cost doubled", dict(transport_rate=2 * cfg.TRANSPORT_UGX_PER_KG_KM)),
        ("no delay before selling at another market", dict(alt_lag_months=0)),
    ]:
        sens.append(headline(AE.evaluate(panel, markets, registry, verbose=False, **kw), label))
    sens = pd.DataFrame(sens)
    sens.to_csv(cfg.ADVICE_EVAL_DIR / "sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    cols = ["crop", "decisions", "advised_store_pct", "advised_other_market_pct",
            "avg_harvest_sale_ugx", "timing_advice_gain_pct", "full_advice_gain_pct",
            "full_advice_better_pct", "full_advice_worse_pct",
            "store_3_gain_pct", "store_6_gain_pct", "hindsight_gain_pct"]
    print(f"\nHarvest months, per {cfg.EVAL_QUANTITY_KG:,} kg (gains are % of the harvest-sale earnings)")
    print(res["summary_harvest"][cols].to_string(index=False))
    print("\nAll crops together")
    print(res["summary_harvest_all_crops"][cols[1:]].to_string(index=False))
    print("\nBefore vs during the test period (harvest months)")
    print(res["summary_by_period"][["period"] + cols[1:]].to_string(index=False))
    print("\nConfidence check (timing advice)")
    print(res["calibration"].to_string(index=False))
    print("\nWorst cases (full advice, harvest months)")
    print(res["worst_cases"].to_string(index=False))
    print("\nSensitivity (harvest months, all crops)")
    print(sens.to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Files in {cfg.ADVICE_EVAL_DIR}")


if __name__ == "__main__":
    main()
