"""
Rebuild every result from the raw WFP CSV, in order. Run from the project root:

    python scripts/run_pipeline.py                  # data, backtest, model choice, advice test, tests
    python scripts/run_pipeline.py --notebook       # also re-run the notebook (adds a few minutes)
    python scripts/run_pipeline.py --demo-providers # also add the fictional demo providers
    python scripts/run_pipeline.py --skip-backtest  # reuse the saved backtest (it takes ~10 minutes)

Steps
  1. build_dataset.py      clean data and monthly panel          -> processed/
  2. run_backtest.py       walk-forward test of every forecaster -> processed/backtest/
  3. select_models.py      method and bands per crop             -> processed/model_registry.json
  4. evaluate_advice.py    advice tested on past prices          -> processed/advice_eval/
  5. pytest                all tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nStopped: '{label}' failed (exit code {result.returncode}).")
        sys.exit(result.returncode)
    print(f"--- {label}: done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--notebook", action="store_true")
    ap.add_argument("--demo-providers", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    t0 = time.time()

    run("1. Build dataset", [py, "scripts/build_dataset.py"])
    if not args.skip_backtest:
        run("2. Walk-forward backtest", [py, "scripts/run_backtest.py", "--test"])
    run("3. Select models", [py, "scripts/select_models.py"])
    run("4. Evaluate advice", [py, "scripts/evaluate_advice.py"])
    if args.notebook:
        run("Notebook", [py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                         "--inplace", "AgriPrice_Advisor_EDA_and_Modeling.ipynb"])
    if args.demo_providers:
        run("Demo providers", [py, "scripts/seed_demo_providers.py"])
    run("5. Tests", [py, "-m", "pytest", "-q", "tests"])
    print(f"\nAll steps finished in {time.time() - t0:.0f}s. Start the app with: "
          "streamlit run app/app.py")


if __name__ == "__main__":
    main()
