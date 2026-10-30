"""
Choose one forecasting method per crop and save processed/model_registry.json.

Run from the project root after scripts/run_backtest.py:

    python scripts/select_models.py

The choice uses the validation split only. The test split is used just to
report how often real prices fell inside the forecast band.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src.forecasting import build_registry  # noqa: E402


def main() -> None:
    reg = build_registry(write=True)
    level = int(reg["interval_level"] * 100)
    print(f"Selection rule: {reg['selection_rule']}")
    print(f"Candidates: {', '.join(reg['candidates'])}\n")
    for crop, e in reg["crops"].items():
        print(f"{crop:<14} -> {e['model']:<15} {e['reason']}")
        cov = [e["horizons"][str(h)]["test_band_coverage"] for h in cfg.HORIZONS]
        mape = [e["horizons"][str(h)]["validation_MAPE"] for h in cfg.HORIZONS]
        print(f"{'':<17} validation MAPE by horizon: {mape}")
        print(f"{'':<17} test coverage of {level}% band:  {cov}")
    print(f"\nSaved {cfg.MODEL_REGISTRY_JSON}")


if __name__ == "__main__":
    main()
