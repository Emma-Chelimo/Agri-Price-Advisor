"""
Data pipeline for AgriPrice Advisor.

Steps
-----
1. load_raw()            read the WFP CSV
2. clean_prices()        apply scope rules, fix regions, resolve duplicates
3. build_monthly_panel() put every retail series on a complete monthly
                         calendar, fill short gaps, mark long ones
4. summarise_series()    one row per series with coverage and staleness
5. build_all()           run everything and write the processed files

Dates are stored as the first day of each month everywhere in the pipeline.
WFP records them on the 15th, which broke pandas' month-start frequency
("MS") and made the SARIMA demo fail on an empty series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)

SERIES_KEYS = ["commodity", "market", "pricetype"]


# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------
def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else cfg.find_raw_csv()
    df = pd.read_csv(path)
    required = {
        "date", "admin1", "market", "market_id", "latitude", "longitude",
        "commodity", "unit", "priceflag", "pricetype", "currency", "price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {sorted(missing)}")
    return df


# -----------------------------------------------------------------------------
# 2. Clean
# -----------------------------------------------------------------------------
@dataclass
class CleaningLog:
    """Row counts at each cleaning step, for the report and for debugging."""
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, step: str, n_rows: int) -> None:
        self.steps.append((step, n_rows))

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.steps, columns=["step", "rows_remaining"])
        out["rows_removed"] = (-out["rows_remaining"].diff()).fillna(0).astype(int)
        return out


def map_region(district: str) -> str:
    try:
        return cfg.REGION_BY_DISTRICT[district]
    except KeyError:
        raise KeyError(
            f"District '{district}' has no region in config.REGION_BY_DISTRICT. "
            "Add it there before rebuilding the data."
        ) from None


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Apply the project's scope and quality rules to the raw WFP data.

    Returns one row per commodity, market, price type and month, with both
    retail and wholesale kept (modelling later selects retail).
    """
    clog = CleaningLog()
    df = raw.copy()
    clog.add("raw rows", len(df))

    # Dates: WFP uses DD/MM/YYYY on the 15th. Convert to the month's first day.
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["period"] = df["date"].dt.to_period("M")
    df["date"] = df["period"].dt.to_timestamp()
    clog.add("valid dates", len(df))

    # Target crops only, mapped to their canonical names.
    df = df[df["commodity"].isin(cfg.COMMODITY_SOURCES)].copy()
    df["source_commodity"] = df["commodity"]
    df["source_priority"] = df["commodity"].map(lambda c: cfg.COMMODITY_SOURCES[c][1])
    df["commodity"] = df["commodity"].map(lambda c: cfg.COMMODITY_SOURCES[c][0])
    clog.add("target crops only", len(df))

    df = df[df["priceflag"] == "actual"]
    clog.add("actual prices only", len(df))

    df = df[(df["unit"] == cfg.EXPECTED_UNIT) & (df["currency"] == cfg.EXPECTED_CURRENCY)]
    clog.add("UGX per KG only", len(df))

    df = df[~df["market"].str.contains(cfg.REFUGEE_MARKER, regex=False, na=False)]
    clog.add("refugee settlement markets removed", len(df))

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]
    clog.add("positive prices only", len(df))

    df["region"] = df["admin1"].map(map_region)

    # Resolve duplicates within a month: keep the highest-priority source
    # (e.g. Maize (white) over plain Maize), then average any remaining ties.
    best = df.groupby(SERIES_KEYS + ["period"])["source_priority"].transform("min")
    df = df[df["source_priority"] == best]
    agg = {
        "price": "mean",
        "usdprice": "mean",
        "date": "first",
        "admin1": "first",
        "admin2": "first",
        "market_id": "first",
        "latitude": "first",
        "longitude": "first",
        "region": "first",
        "source_commodity": "first",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    df = df.groupby(SERIES_KEYS + ["period"], as_index=False).agg(agg)
    clog.add("one row per series-month", len(df))

    cols = [
        "date", "period", "commodity", "source_commodity", "market", "market_id",
        "admin1", "admin2", "region", "latitude", "longitude", "pricetype",
        "price", "usdprice",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    return df, clog


# -----------------------------------------------------------------------------
# 3. Monthly panel on a complete calendar
# -----------------------------------------------------------------------------
def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each True position, the length of the run of Trues it belongs to."""
    out = np.zeros(len(mask), dtype=int)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def _complete_series(s: pd.DataFrame) -> pd.DataFrame:
    """Reindex one series to every month between its first and last price."""
    s = s.set_index("period").sort_index()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    out = pd.DataFrame(index=full)
    out.index.name = "period"
    out["price_observed"] = s["price"]

    missing = out["price_observed"].isna().to_numpy()
    run_len = _run_lengths(missing)
    short_gap = missing & (run_len <= cfg.MAX_INTERPOLATE_GAP)
    long_gap = missing & (run_len > cfg.MAX_INTERPOLATE_GAP)

    # Interpolate in log space so a price moving 1000 -> 2000 fills
    # proportionally rather than in equal shilling steps.
    log_p = np.log(out["price_observed"])
    log_filled = log_p.interpolate(method="linear", limit_area="inside")
    price = out["price_observed"].copy()
    price[short_gap] = np.exp(log_filled[short_gap])

    out["price"] = price
    out["is_imputed"] = short_gap
    out["in_long_gap"] = long_gap
    out["gap_run_length"] = np.where(missing, run_len, 0)

    # A new segment starts after every long gap.
    starts_long = long_gap & ~np.r_[False, long_gap[:-1]]
    out["segment_id"] = np.cumsum(starts_long)

    # Month-on-month log change.
    r = np.log(out["price"]).diff()
    r[out["in_long_gap"] | out["in_long_gap"].shift(1, fill_value=False)] = np.nan
    out["mom_log_change"] = r

    # Spike flag (flag only, never removed): a large jump that reverses by at
    # least half the next month. Large moves that persist are treated as real
    # market moves, since WFP prices often shift 20-30% in a month.
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    threshold = max(cfg.OUTLIER_Z * (mad if np.isfinite(mad) else 0.0),
                    cfg.OUTLIER_MIN_JUMP)
    nxt = r.shift(-1)
    spike = (r.abs() > threshold) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= 0.5 * r.abs())
    out["is_outlier_flag"] = spike.fillna(False).astype(bool)

    # Static attributes of the market.
    for col in ["commodity", "market", "pricetype", "market_id", "admin1",
                "region", "latitude", "longitude"]:
        out[col] = s[col].iloc[0]
    return out.reset_index()


def build_monthly_panel(
    cleaned: pd.DataFrame,
    pricetype: str = cfg.MODEL_PRICE_TYPE,
    min_observed: int = cfg.MIN_OBSERVED_MONTHS,
) -> pd.DataFrame:
    """
    One row per series and month with no missing months in between.

    Columns of note
    ---------------
    price            observed price, or interpolated for short gaps,
                     NaN inside long gaps
    price_observed   the raw price, NaN for every month WFP did not record
    is_imputed       True where price was interpolated
    in_long_gap      True inside a gap longer than MAX_INTERPOLATE_GAP
    segment_id       0, 1, 2 ... increases after every long gap; lag features
                     should be built within a segment only
    is_outlier_flag  suspiciously large month-on-month move, for review
    """
    data = cleaned[cleaned["pricetype"] == pricetype]
    n_obs = data.groupby(SERIES_KEYS).size()
    keep = n_obs[n_obs >= min_observed].index
    data = data.set_index(SERIES_KEYS).loc[keep].reset_index()

    parts = [_complete_series(g) for _, g in data.groupby(SERIES_KEYS, sort=True)]
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = panel["period"].dt.to_timestamp()

    cols = [
        "commodity", "market", "pricetype", "region", "admin1", "market_id",
        "latitude", "longitude", "period", "date", "price", "price_observed",
        "is_imputed", "in_long_gap", "gap_run_length", "segment_id",
        "mom_log_change", "is_outlier_flag",
    ]
    return panel[cols].sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 4. Series summary, staleness and latest prices
# -----------------------------------------------------------------------------
def summarise_series(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series with coverage and staleness.

    Staleness is measured against the newest month in the whole dataset, not
    today's date, so results do not change just because time has passed.
    """
    reference = panel.loc[panel["price_observed"].notna(), "period"].max()
    rows = []
    for key, g in panel.groupby(SERIES_KEYS, sort=True):
        obs = g[g["price_observed"].notna()]
        last = obs.iloc[-1]
        latest_seg = g[g["segment_id"] == g["segment_id"].max()]
        months_since = (reference - last["period"]).n
        rows.append({
            "commodity": key[0],
            "market": key[1],
            "pricetype": key[2],
            "region": g["region"].iloc[0],
            "first_month": g["period"].min().to_timestamp(),
            "last_observed_month": last["period"].to_timestamp(),
            "span_months": len(g),
            "n_observed": len(obs),
            "n_imputed": int(g["is_imputed"].sum()),
            "n_long_gap_months": int(g["in_long_gap"].sum()),
            "n_segments": int(g["segment_id"].max()) + 1,
            "latest_segment_months": int(latest_seg["price"].notna().sum()),
            "coverage_pct": round(100 * len(obs) / len(g), 1),
            "n_outlier_flags": int(g["is_outlier_flag"].sum()),
            "last_price": float(last["price_observed"]),
            "months_since_last_obs": months_since,
            "is_stale": months_since > cfg.STALE_AFTER_MONTHS,
        })
    out = pd.DataFrame(rows)
    out.attrs["reference_month"] = str(reference)
    return out


def latest_prices(panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Most recent observed price per series, with its age and stale flag."""
    cols = SERIES_KEYS + ["region", "last_observed_month", "last_price",
                          "months_since_last_obs", "is_stale"]
    return summary[cols].rename(columns={
        "last_observed_month": "date", "last_price": "price",
    })


def market_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per market with its coordinates, used for distances later."""
    return (cleaned.groupby("market", as_index=False)
            .agg(market_id=("market_id", "first"),
                 admin1=("admin1", "first"),
                 region=("region", "first"),
                 latitude=("latitude", "first"),
                 longitude=("longitude", "first"))
            .sort_values(["region", "market"])
            .reset_index(drop=True))


# -----------------------------------------------------------------------------
# 5. Build everything
# -----------------------------------------------------------------------------
def build_all(raw_path: Path | None = None, write: bool = True) -> dict:
    raw = load_raw(raw_path)
    cleaned, clog = clean_prices(raw)
    panel = build_monthly_panel(cleaned)
    summary = summarise_series(panel)
    latest = latest_prices(panel, summary)
    markets = market_table(cleaned)

    if write:
        cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Periods do not round-trip cleanly through parquet; store dates only.
        cleaned.drop(columns="period").to_parquet(cfg.CLEANED_PARQUET, index=False)
        panel.drop(columns="period").to_parquet(cfg.MONTHLY_PARQUET, index=False)
        summary.to_csv(cfg.SERIES_SUMMARY_CSV, index=False)
        latest.to_csv(cfg.LATEST_PRICES_CSV, index=False)
        markets.to_csv(cfg.MARKETS_CSV, index=False)

    return {
        "raw": raw, "cleaned": cleaned, "panel": panel, "summary": summary,
        "latest": latest, "markets": markets, "cleaning_log": clog.to_frame(),
    }


# -----------------------------------------------------------------------------
# Loaders for the notebook and the app
# -----------------------------------------------------------------------------
def load_monthly(path: Path | None = None) -> pd.DataFrame:
    """Read the monthly panel and restore the period column."""
    df = pd.read_parquet(path or cfg.MONTHLY_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period("M")
    return df


def load_series_summary(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cfg.SERIES_SUMMARY_CSV,
                       parse_dates=["first_month", "last_observed_month"])


def load_markets(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cfg.MARKETS_CSV)


def get_series(panel: pd.DataFrame, commodity: str, market: str,
               pricetype: str = cfg.MODEL_PRICE_TYPE,
               latest_segment_only: bool = False) -> pd.DataFrame:
    """Rows for one series, optionally only the segment after the last long gap."""
    s = panel[(panel["commodity"] == commodity)
              & (panel["market"] == market)
              & (panel["pricetype"] == pricetype)].sort_values("date")
    if latest_segment_only and not s.empty:
        s = s[s["segment_id"] == s["segment_id"].max()]
    return s
