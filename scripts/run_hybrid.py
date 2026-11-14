"""
Backtest the SARIMA + LSTM + XGBoost hybrid against the current system.

    python scripts/run_hybrid.py

Writes processed/hybrid/: sarima_table.parquet, residual_predictions.parquet,
forecasts.parquet (hybrid variants on the same pairs as the main backtest).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np, pandas as pd
from src import config as cfg, data, hybrid as HY, pooled as P

out = cfg.PROCESSED_DIR / "hybrid"; out.mkdir(parents=True, exist_ok=True)
panel = data.load_monthly()
t0 = time.time()
sp = out / "sarima_table.parquet"
if sp.exists():
    sar = pd.read_parquet(sp)
else:
    sar = HY.sarima_table(panel, order="A")
    sar.to_parquet(sp, index=False)
print(f"SARIMA table: {len(sar)} origins, {time.time()-t0:.0f}s", flush=True)
t = HY.attach_residuals(sar, panel)
feats = P.build_features(panel)
fit_dates = list(pd.date_range(cfg.CALIBRATION_START + "-01", t["origin"].max(), freq="QS"))
t = HY.run_residual_models(t, feats, fit_dates)
t.to_parquet(out / "residual_predictions.parquet", index=False)
fc = HY.to_forecasts(t)
fc.to_parquet(out / "hybrid_forecasts_raw.parquet", index=False)
print(f"Done in {time.time()-t0:.0f}s")
