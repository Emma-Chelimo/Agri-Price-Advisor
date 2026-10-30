"""
One call that produces everything a farmer's advice screen shows.

    result = get_advice(crop="Millet", market="Mbarara", quantity_kg=500)

returns the forecast, the sell-or-store advice, the market comparison, the
template explanation and the payload the language model works from. The web
API (api/) and any other interface use this, so the figures are computed in
exactly one place, on the server.

Data files are loaded once and kept in memory (load_context). Call
reload_context() after rebuilding the data or choosing new models.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import assistant, decision, forecasting
from . import config as cfg
from . import data as pipeline


# -----------------------------------------------------------------------------
# Shared data, loaded once
# -----------------------------------------------------------------------------
@dataclass
class Context:
    panel: pd.DataFrame
    summary: pd.DataFrame
    markets: pd.DataFrame
    registry: dict


_CTX: Context | None = None


def load_context() -> Context:
    global _CTX
    if _CTX is None:
        if not cfg.MONTHLY_PARQUET.exists():
            pipeline.build_all(write=True)
        _CTX = Context(panel=pipeline.load_monthly(), summary=pipeline.load_series_summary(),
                       markets=pipeline.load_markets(), registry=forecasting.load_registry())
    return _CTX


def reload_context() -> Context:
    global _CTX
    _CTX = None
    _cached_advice.cache_clear()
    return load_context()


# -----------------------------------------------------------------------------
# Advice
# -----------------------------------------------------------------------------
@dataclass
class AdviceResult:
    crop: str
    market: str
    quantity_kg: float
    horizon: int
    series: pd.DataFrame                  # price history used for the chart
    fc: forecasting.Forecast
    costs: decision.Costs
    adv: decision.Advice
    mc: decision.MarketComparison
    alt: pd.Series | None
    months_old: int | None
    is_stale: bool

    @property
    def go_to_alternative(self) -> bool:
        return self.alt is not None and self.alt["best_net_ugx"] > self.adv.expected_net_ugx

    @property
    def sell_market(self) -> str:
        return self.alt["market"] if self.go_to_alternative else self.market

    def payload(self) -> dict:
        """The figures the language model may use (see src/assistant.py)."""
        return assistant.build_payload(
            crop=self.crop, market=self.market, quantity_kg=self.quantity_kg, fc=self.fc,
            adv=self.adv, costs=self.costs, mc=self.mc, alt=self.alt,
            months_old=self.months_old, is_stale=self.is_stale)

    def template_explanation(self) -> str:
        return template_explanation(self)


def _costs(crop: str, storage: float | None, loss_pct: float | None,
           transport_rate: float | None) -> decision.Costs:
    c = decision.Costs.default_for(crop)
    if storage is not None:
        c.storage_ugx_per_kg_month = float(storage)
    if loss_pct is not None:
        c.loss_rate_per_month = float(loss_pct) / 100
    if transport_rate is not None:
        c.transport_ugx_per_kg_km = float(transport_rate)
    return c


@lru_cache(maxsize=256)
def _cached_advice(crop, market, quantity_kg, horizon, storage, loss_pct, transport_rate):
    ctx = load_context()
    horizon = int(max(1, min(horizon, cfg.MAX_FORECAST_HORIZON)))
    series = pipeline.get_series(ctx.panel, crop, market)
    series = series[series["price"].notna()]
    if series.empty:
        raise ValueError(f"No {crop} price history for {market}.")
    costs = _costs(crop, storage, loss_pct, transport_rate)
    fc = forecasting.forecast(ctx.panel, crop, market, horizon, ctx.registry)
    adv = decision.advise(fc, quantity_kg, costs)
    mc = decision.compare_markets(ctx.panel, ctx.markets, crop, market, quantity_kg, horizon,
                                  costs, ctx.registry)
    alt = decision.best_alternative(mc)
    row = ctx.summary[(ctx.summary["commodity"] == crop) & (ctx.summary["market"] == market)]
    months_old = int(row["months_since_last_obs"].iloc[0]) if len(row) else None
    is_stale = bool(len(row) and row["is_stale"].iloc[0])
    return AdviceResult(crop, market, float(quantity_kg), horizon, series, fc, costs, adv, mc, alt,
                        months_old, is_stale)


def get_advice(*, crop: str, market: str, quantity_kg: float,
               horizon: int = cfg.MAX_FORECAST_HORIZON, storage: float | None = None,
               loss_pct: float | None = None, transport_rate: float | None = None) -> AdviceResult:
    if quantity_kg <= 0:
        raise ValueError("Quantity must be greater than zero.")
    return _cached_advice(crop, market, float(quantity_kg), int(horizon),
                          None if storage is None else float(storage),
                          None if loss_pct is None else float(loss_pct),
                          None if transport_rate is None else float(transport_rate))


# -----------------------------------------------------------------------------
# Template explanation (used when no language model is available)
# -----------------------------------------------------------------------------
def _pct(p: float | None) -> str:
    return "n/a" if p is None or not np.isfinite(p) else f"{p * 100:.0f}%"


def template_explanation(r: AdviceResult) -> str:
    """Plain explanation built only from computed numbers. Markdown bold allowed."""
    adv, fc, costs, alt = r.adv, r.fc, r.costs, r.alt
    parts = []
    if adv.months == 0:
        parts.append(
            f"Selling your {r.quantity_kg:,.0f} kg of **{r.crop}** at **{r.market}** now, at "
            f"{fc.origin_price:,.0f} UGX/kg, gives about **{adv.net_sell_now_ugx:,.0f} UGX**.")
        if adv.best_store_months:
            opt = adv.options.set_index("months").loc[adv.best_store_months]
            parts.append(
                f"Storing only pays if the price rises enough to cover storage "
                f"({costs.storage_ugx_per_kg_month:,.0f} UGX/kg a month) and losses "
                f"({costs.loss_rate_per_month * 100:.2f}% a month). The best storage option, "
                f"{opt['action'].lower()}, needs at least **{opt['breakeven_price']:,.0f} UGX/kg**, "
                f"and the chance of reaching that is about **{_pct(adv.best_store_prob)}**.")
    else:
        parts.append(
            f"**{adv.recommendation}** is expected to earn about **{adv.gain_ugx:,.0f} UGX** more "
            f"than selling your {r.quantity_kg:,.0f} kg of {r.crop} at {r.market} now, after storage "
            f"costs and losses. The chance it beats selling now is about **{_pct(adv.confidence)}**.")
    parts.append(f"Confidence: **{adv.confidence_label}** ({_pct(adv.confidence)}).")
    if fc.model == "naive":
        parts.append(
            "The forecast assumes the price stays at its latest recorded level, because on past "
            "data no method has yet predicted these prices better than that. The chances above "
            "come from how far real prices strayed from such forecasts before.")
    else:
        parts.append(f"Forecast method: {fc.label}, chosen because it beat the simple baseline "
                     "on past data.")
    if r.go_to_alternative:
        parts.append(
            f"**{alt['market']}** ({alt['road_km']:,.0f} km away) could pay about "
            f"{alt['best_net_ugx'] - adv.expected_net_ugx:,.0f} UGX more even after "
            f"{alt['transport_ugx']:,.0f} UGX of transport.")
    return "\n\n".join(parts)
