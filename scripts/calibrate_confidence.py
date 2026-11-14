"""
Make the advice's confidence figure mean what it says.

Run from the project root after scripts/evaluate_advice.py:

    python scripts/calibrate_confidence.py

The advice evaluation replays past decisions and records the raw confidence
(the model's probability) and whether the advice turned out right. Raw
probabilities were overstated: "sell now" advice stated about 54% and was
right about 43% of the time. This script fits an isotonic (never decreasing)
map from raw probability to the share of advice that was right, separately for
"sell now" and "store" advice, using ONLY decisions before the test period,
then checks it on the test-period decisions.

Writes processed/confidence_calibration.json (read by src/decision.py) and
processed/advice_eval/calibration_check.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from src import config as cfg  # noqa: E402


def main() -> None:
    d = pd.read_parquet(cfg.ADVICE_EVAL_DIR / "decisions.parquet")
    raw = "confidence_raw" if "confidence_raw" in d else "confidence"
    d = d[d[raw].notna() & d["advice_right"].notna()].copy()
    d["kind"] = np.where(d["advice_months"] > 0, "store", "sell")

    cal_map, rows = {}, []
    for kind in ("sell", "store"):
        tr = d[(d["period"] == "pre-test") & (d["kind"] == kind)]
        te = d[(d["period"] == "test") & (d["kind"] == kind)]
        iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
        iso.fit(tr[raw], tr["advice_right"])
        cal_map[kind] = {"x": [round(float(v), 4) for v in iso.X_thresholds_],
                         "y": [round(float(v), 4) for v in iso.y_thresholds_],
                         "fitted_on_decisions": int(len(tr))}
        for period, part in (("pre-test (fitted)", tr), ("test (check)", te)):
            rows.append({"advice": kind, "period": period, "decisions": len(part),
                         "raw_stated_pct": round(100 * part[raw].mean(), 1),
                         "calibrated_stated_pct": round(100 * iso.predict(part[raw]).mean(), 1),
                         "actually_right_pct": round(100 * part["advice_right"].mean(), 1)})

    cfg.CONFIDENCE_CALIBRATION_JSON.write_text(json.dumps(
        {"fitted_on": f"advice decisions before {cfg.TEST_START}", **cal_map}, indent=2))
    out = pd.DataFrame(rows)
    out.to_csv(cfg.ADVICE_EVAL_DIR / "calibration_check.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {cfg.CONFIDENCE_CALIBRATION_JSON}")


if __name__ == "__main__":
    main()
