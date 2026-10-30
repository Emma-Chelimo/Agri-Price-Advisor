"""
Logic behind the app's user flow (Phase 6), kept out of the UI so it can be
tested on its own.

    normalise_ug_phone()   accept 07XX..., +2567XX..., 2567XX... and spaces
    available_crops()      crops with at least one market that has recent data
    available_markets()    markets with recent data for a crop
    nearest_market()       nearest market with recent data to a GPS position
    resolve_market()       which market to use: manual choice, else nearest
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .decision import haversine_km


# -----------------------------------------------------------------------------
# Phone numbers
# -----------------------------------------------------------------------------
def normalise_ug_phone(raw: str) -> str | None:
    """
    Return the number as +2567XXXXXXXX, or None if it is not a valid Ugandan
    mobile number. Accepts spaces, dashes and brackets, and the forms
    07XXXXXXXX, 7XXXXXXXX, 2567XXXXXXXX, +2567XXXXXXXX and 002567XXXXXXXX.
    """
    if not raw:
        return None
    s = re.sub(r"[\s\-().]", "", raw.strip())
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("00"):
        s = s[2:]
    if s.startswith("256"):
        s = s[3:]
    elif s.startswith("0"):
        s = s[1:]
    if re.fullmatch(r"7\d{8}", s):
        return cfg.UG_COUNTRY_CODE + s
    return None


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")


def normalise_email(raw: str) -> str | None:
    """Return the address trimmed and lower-cased, or None if it is not valid."""
    if not raw:
        return None
    s = raw.strip().lower()
    if len(s) > 254 or ".." in s or not _EMAIL_RE.fullmatch(s):
        return None
    return s


def parse_contact(raw: str) -> tuple[str, str] | None:
    """
    Sign-in accepts a phone number or an email address in one field.
    Returns ("phone", "+2567XXXXXXXX") or ("email", "name@example.com"),
    or None if the text is neither.
    """
    if raw and "@" in raw:
        email = normalise_email(raw)
        return ("email", email) if email else None
    phone = normalise_ug_phone(raw)
    return ("phone", phone) if phone else None


# -----------------------------------------------------------------------------
# What the farmer can choose from
# -----------------------------------------------------------------------------
def recent_series(summary: pd.DataFrame) -> pd.DataFrame:
    """Retail series whose latest price is not stale (config.STALE_AFTER_MONTHS)."""
    return summary[(summary["pricetype"] == cfg.MODEL_PRICE_TYPE) & (~summary["is_stale"])]


def available_crops(summary: pd.DataFrame) -> list[str]:
    crops = set(recent_series(summary)["commodity"])
    return [c for c in cfg.TARGET_CROPS if c in crops]


def available_markets(summary: pd.DataFrame, crop: str) -> list[str]:
    s = recent_series(summary)
    return sorted(s.loc[s["commodity"] == crop, "market"].unique())


# -----------------------------------------------------------------------------
# Location
# -----------------------------------------------------------------------------
@dataclass
class NearestMarket:
    market: str | None
    distance_km: float | None      # straight-line distance
    in_coverage: bool


def nearest_market(lat: float, lon: float, markets: pd.DataFrame,
                   candidates: list[str]) -> NearestMarket:
    """Nearest of `candidates` to (lat, lon); out of coverage if too far."""
    m = markets[markets["market"].isin(candidates)]
    if m.empty:
        return NearestMarket(None, None, False)
    d = haversine_km(lat, lon, m["latitude"].to_numpy(), m["longitude"].to_numpy())
    i = int(np.argmin(d))
    km = float(d[i])
    return NearestMarket(m["market"].iloc[i], round(km, 1), km <= cfg.MAX_LOCATION_TO_MARKET_KM)


@dataclass
class MarketChoice:
    market: str | None
    source: str                   # "manual", "location" or "none"
    distance_km: float | None = None
    note: str = ""


def resolve_market(crop: str, summary: pd.DataFrame, markets: pd.DataFrame,
                   coords: tuple[float, float] | None,
                   manual: str | None) -> MarketChoice:
    """
    Pick the farmer's market for this crop.

    1. A market the farmer chose (in settings or on the crop page), if it has
       recent prices for this crop.
    2. Otherwise the nearest market with recent prices for this crop, if the
       detected location is inside coverage.
    3. Otherwise none, and the app asks the farmer to choose.
    """
    options = available_markets(summary, crop)
    note = ""
    if manual:
        if manual in options:
            return MarketChoice(manual, "manual")
        note = f"{manual} has no recent {crop} prices, so the nearest market with them is used."
    if coords is not None:
        nm = nearest_market(coords[0], coords[1], markets, options)
        if nm.market and nm.in_coverage:
            return MarketChoice(nm.market, "location", nm.distance_km, note)
        if nm.market:
            note = (f"Your location is about {nm.distance_km:,.0f} km from the nearest market "
                    "with price data, so please choose your market.")
    return MarketChoice(None, "none", None, note)
