"""
Project-wide configuration for AgriPrice Advisor.

This is the single source of truth for paths, scope and data rules.
The notebook, the build script and the Streamlit app all import from here,
so a change made once applies everywhere.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (portable across Windows, macOS and Linux)
# -----------------------------------------------------------------------------
# config.py lives in <project>/src/, so the project root is one level up.
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"

RAW_CSV_NAME = "wfp_food_prices_uga.csv"


def find_raw_csv() -> Path:
    """Return the WFP price CSV, accepting older file names as a fallback."""
    preferred = ROOT / RAW_CSV_NAME
    if preferred.exists():
        return preferred
    candidates = sorted(ROOT.glob("wfp_food_prices*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No WFP price CSV found in {ROOT}. Expected '{RAW_CSV_NAME}'."
    )


# Output files written by scripts/build_dataset.py
CLEANED_PARQUET = PROCESSED_DIR / "cleaned_prices.parquet"
MONTHLY_PARQUET = PROCESSED_DIR / "monthly_prices.parquet"
SERIES_SUMMARY_CSV = PROCESSED_DIR / "series_summary.csv"
LATEST_PRICES_CSV = PROCESSED_DIR / "latest_prices.csv"
MARKETS_CSV = PROCESSED_DIR / "markets.csv"

# -----------------------------------------------------------------------------
# Scope
# -----------------------------------------------------------------------------
# The six staple crops named in the proposal. Fresh cassava is out of scope.
TARGET_CROPS = [
    "Maize",
    "Maize flour",
    "Beans",
    "Sorghum",
    "Millet",
    "Cassava flour",
]

# The raw file holds two maize series. "Maize (white)" is the main retail
# series. Plain "Maize" is mostly wholesale, and where both exist for the same
# market and month the retail prices differ by about 23% on average, so they
# are not averaged together. The plain series only fills months that the main
# series is missing. Lower number = higher priority.
COMMODITY_SOURCES = {
    "Maize (white)": ("Maize", 0),
    "Maize": ("Maize", 1),
    "Maize flour": ("Maize flour", 0),
    "Beans": ("Beans", 0),
    "Sorghum": ("Sorghum", 0),
    "Millet": ("Millet", 0),
    "Cassava flour": ("Cassava flour", 0),
}

# Modelling uses retail prices only. Wholesale data stops in 2022 and covers
# few markets. Cleaned data keeps both types for exploratory analysis.
MODEL_PRICE_TYPE = "Retail"

# Refugee settlement markets are excluded from scope. WFP marks all 13 of them
# with this suffix in the market name.
REFUGEE_MARKER = "(refugee settlement)"

# Only keep prices recorded per kilogram in Ugandan shillings.
EXPECTED_UNIT = "KG"
EXPECTED_CURRENCY = "UGX"

# -----------------------------------------------------------------------------
# Regions
# -----------------------------------------------------------------------------
# District (WFP 'admin1') to region, following the four UBOS statistical
# regions. Karamoja districts fall under Northern in the UBOS four-region split.
# Corrections from the old map: Soroti is Eastern (Teso), Kiryandongo is
# Western (Bunyoro).
REGION_BY_DISTRICT = {
    # Central
    "Kampala": "Central",
    "Kyankwanzi": "Central",
    "Masaka": "Central",
    "Mukono": "Central",
    "Wakiso": "Central",
    # Eastern
    "Busia": "Eastern",
    "Iganga": "Eastern",
    "Jinja": "Eastern",
    "Kween": "Eastern",
    "Mbale": "Eastern",
    "Soroti": "Eastern",
    "Tororo": "Eastern",
    # Northern (including Karamoja and West Nile)
    "Abim": "Northern",
    "Adjumani": "Northern",
    "Amudat": "Northern",
    "Arua": "Northern",
    "Gulu": "Northern",
    "Kaabong": "Northern",
    "Koboko": "Northern",
    "Kotido": "Northern",
    "Lamwo": "Northern",
    "Lira": "Northern",
    "Moroto": "Northern",
    "Moyo": "Northern",
    "Nakapiripirit": "Northern",
    "Napak": "Northern",
    "Yumbe": "Northern",
    # Western
    "Hoima": "Western",
    "Isingiro": "Western",
    "Kabale": "Western",
    "Kabarole": "Western",
    "Kamwenge": "Western",
    "Kasese": "Western",
    "Kiryandongo": "Western",
    "Kyegegwa": "Western",
    "Masindi": "Western",
    "Mbarara": "Western",
}

REGIONS = ["Central", "Eastern", "Northern", "Western"]

# -----------------------------------------------------------------------------
# Monthly calendar and gap handling
# -----------------------------------------------------------------------------
# Gaps of this many missing months or fewer are filled by interpolation
# (linear in log price). Longer gaps stay missing and split the series into
# separate segments, so lag features never reach across them.
MAX_INTERPOLATE_GAP = 2

# A series needs at least this many observed (not interpolated) months to be
# kept in the monthly panel.
MIN_OBSERVED_MONTHS = 36

# A series counts as stale when its most recent observed price is more than
# this many months older than the newest month in the whole dataset.
STALE_AFTER_MONTHS = 6

# Possible data errors are flagged (never removed) when a month's price jumps
# and then reverses by at least half the next month. The jump must exceed both
# OUTLIER_Z robust standard deviations of the series' monthly changes and
# OUTLIER_MIN_JUMP in log terms (0.405 is roughly a 50% move).
OUTLIER_Z = 5.0
OUTLIER_MIN_JUMP = 0.405

# -----------------------------------------------------------------------------
# Evaluation (walk-forward backtest)
# -----------------------------------------------------------------------------
# Forecast horizons in months. The app's sell-or-store advice looks up to six
# months ahead, so every model is judged at each of these horizons.
HORIZONS = [1, 2, 3, 4, 5, 6]

# Calendar cutoffs shared by every series, so a pooled model trained across
# markets can never learn from a month that another series is tested on.
#   validation: forecast origins from VALIDATION_START up to TEST_START, and
#               only targets before TEST_START count. Use this for tuning and
#               early stopping.
#   test:       forecast origins from TEST_START onward. Touch it only to
#               report final results.
VALIDATION_START = "2022-01"
TEST_START = "2024-01"

# A forecast origin needs at least this many observed months of history.
MIN_TRAIN_MONTHS = 24

BACKTEST_DIR = PROCESSED_DIR / "backtest"

# -----------------------------------------------------------------------------
# Model selection and serving (Phase 4)
# -----------------------------------------------------------------------------
# Written by scripts/select_models.py, read by the app.
MODEL_REGISTRY_JSON = PROCESSED_DIR / "model_registry.json"

# The method every crop falls back to. Phase 2 showed "same as last month"
# beats seasonal-naive at every horizon, so naive is the safe default.
FALLBACK_MODEL = "naive"

# A model replaces naive for a crop only if its average validation skill
# across horizons is at least this much above naive's (0.02 = 2% lower MAE).
# The margin stops a model being picked on noise.
SELECTION_MIN_SKILL = 0.02

# Methods that are never selected for the app. app_baseline is the old
# dashboard forecast, kept in the backtest only as a reference point.
NOT_SELECTABLE = {"app_baseline"}

# Forecast bands shown in the app cover this share of past outcomes.
# They come from the selected model's validation errors for that crop and
# horizon; INTERVAL_MIN_PAIRS is the least data used before falling back to
# the model's errors pooled across all crops.
INTERVAL_LEVEL = 0.80
INTERVAL_MIN_PAIRS = 40

# The app never forecasts further ahead than the backtest has validated.
MAX_FORECAST_HORIZON = max(HORIZONS)

# -----------------------------------------------------------------------------
# Decision logic (Phase 5)
# -----------------------------------------------------------------------------
# Default costs per crop. The farmer can override every value in the app.
#
# storage_loss_pct_per_month
#   APHLIS (African Postharvest Losses Information System) puts Uganda's
#   household-storage losses at 2.6% for maize, 1.4% for sorghum and 0.6% for
#   millet over the storage period (aphlis.net, Uganda 2020). APHLIS does not
#   state the period, so we assume about six months of storage and divide by
#   six. APHLIS has no figures for beans or flours: beans use 1.5% per month
#   because bruchid weevils make them the most storage-sensitive crop here
#   (the proposal cites about 22% harvest and post-harvest loss for beans),
#   and flours use 1.0% as an assumption.
# storage_cost_ugx_per_kg_month
#   No published Ugandan warehouse tariff was found. 20 UGX per kg per month
#   (2,000 UGX per 100 kg bag) is an assumption the farmer should replace with
#   what they actually pay.
CROP_COSTS = {
    "Maize":         {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 0.45},
    "Sorghum":       {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 0.25},
    "Millet":        {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 0.10},
    "Beans":         {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 1.50},
    "Maize flour":   {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 1.00},
    "Cassava flour": {"storage_cost_ugx_per_kg_month": 20, "storage_loss_pct_per_month": 1.00},
}

# Transport to another market, in UGX per kg per km of road.
# Derivation: the TradeMark Africa regional transport study (2022) puts truck
# costs on Kampala-Gulu at about 12.2 USD per km. At roughly 3,700 UGX per USD
# and a 30-tonne load that is about 1.5 UGX per kg per km for a full truck.
# Smallholders usually ship part loads, so the default doubles it.
TRANSPORT_UGX_PER_KG_KM = 3.0

# Straight-line (haversine) distance times this factor approximates road
# distance, since roads rarely run straight between towns.
ROAD_FACTOR = 1.3

# Storing is advised only if the chance that it beats selling now is at least
# this high. Below it the advice is to sell now.
STORE_MIN_PROBABILITY = 0.60

# Confidence labels shown with the advice.
CONFIDENCE_LABELS = [(0.75, "High"), (0.60, "Medium"), (0.0, "Low: a close call")]

# Market comparison only uses markets whose latest price was recorded in the
# same month as the farmer's market (0 = same month only). Comparing prices
# from different months mixes market differences with price changes over time.
COMPARE_MAX_MONTH_GAP = 0

# -----------------------------------------------------------------------------
# User flow (Phase 6)
# -----------------------------------------------------------------------------
# A detected location further than this (straight line) from every market in
# the data is treated as outside coverage, and the farmer picks a market.
MAX_LOCATION_TO_MARKET_KM = 150

# Ugandan mobile numbers: 9 digits after the country code, starting with 7.
UG_COUNTRY_CODE = "+256"

# How many years of history the price chart shows by default (the full
# history is still there; zoom out with the chart controls).
CHART_DEFAULT_YEARS = 3

# -----------------------------------------------------------------------------
# Advice evaluation (Phase 8, Objective 8)
# -----------------------------------------------------------------------------
# Approximate harvest months. Most of Uganda has two seasons: the first-season
# harvest around June to August, the second around November to January (FEWS
# NET Uganda outlooks report second-season harvests running through January
# into early February). Sorghum and millet in the north and Karamoja are
# mostly harvested August to October. Flours are processed products with no
# harvest, so they are evaluated at every month only.
HARVEST_MONTHS = {
    "Maize":   [6, 7, 8, 12, 1],
    "Beans":   [6, 7, 11, 12, 1],
    "Sorghum": [7, 8, 9, 10, 12, 1],
    "Millet":  [7, 8, 9, 10, 12, 1],
}

# Quantity used for the evaluation. Results are reported per 1,000 kg so they
# scale to any amount.
EVAL_QUANTITY_KG = 1000

# A decision point needs this many observed months of history at the
# farmer's market, so the advice is never based on a series that just started.
EVAL_MIN_HISTORY_MONTHS = 12

# When the advice says another market pays more, the farmer only learns that
# market's price from the latest WFP figures, then has to arrange transport.
# The evaluation therefore scores a move at that market's price this many
# months later, not at the price that triggered the move.
EVAL_ALT_MARKET_LAG_MONTHS = 1

ADVICE_EVAL_DIR = PROCESSED_DIR / "advice_eval"

# -----------------------------------------------------------------------------
# Services platform (Phase 9, Objectives 5 and 6)
# -----------------------------------------------------------------------------
# SQLite database for providers, bookings, status updates and ratings. It is
# created on first use. Keep it out of version control: it holds contacts.
DATA_DIR = ROOT / "data"
SERVICES_DB = DATA_DIR / "agriprice.db"
UPLOADS_DIR = DATA_DIR / "uploads"

# Kinds of service and how each prices its work.
SERVICE_KINDS = {
    "storage":      {"label": "Storage",                   "price_unit": "UGX per kg per month"},
    "preservation": {"label": "Preservation (drying, treatment)", "price_unit": "UGX per kg"},
    "transport":    {"label": "Transport",                 "price_unit": "UGX per trip"},
    "buyer":        {"label": "Buyer",                     "price_unit": "UGX per kg paid"},
}

# A provider must reply to a booking request within this many hours, or the
# request expires and the farmer is offered the next ranked provider.
QUOTE_DEADLINE_HOURS = 48

# Shared transport: bookings on the same route whose dates are within this
# many days of each other can share one trip, with the cost split by kg.
SHARED_TRIP_WINDOW_DAYS = 3

# Ranked options shown to the farmer (more on request).
RANKED_OPTIONS_SHOWN = 3
