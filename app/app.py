from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# app.py lives in <project>/app/, so the project root is its parent.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as pipeline  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402
import os  # noqa: E402
import json  # noqa: E402

from src import assistant, decision, flow, forecasting  # noqa: E402

import services_ui  # noqa: E402  (app/services_ui.py: Part 2 screens)

# Browser geolocation (optional). Without it the farmer picks a market once.
try:
    from streamlit_js_eval import get_geolocation
    GEO_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEO_AVAILABLE = False

st.set_page_config(page_title="AgriPrice Advisor", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

STEPS = ["welcome", "crop", "dashboard", "services", "bookings", "provider", "admin"]
ROLE_HOME = {"farmer": "crop", "provider": "provider", "admin": "admin"}
ROLE_STEPS = {"farmer": {"crop", "dashboard", "services", "bookings"},
              "provider": {"provider"}, "admin": {"admin"}}
ROLE_LABELS = {"farmer": "Farmer", "provider": "Service provider or buyer",
               "admin": "Administrator"}
AUTO = "Automatic (nearest to you)"


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
PERSONAL_DEFAULTS = {
    "logged_in": False,
    "role": "farmer",
    "advice_ctx": None,        # the accepted advice, carried into Part 2
    "chosen_provider": None,
    "phone": "",
    "email": "",
    "farmer_name": "",
    "coords": None,            # (lat, lon) from the browser, if allowed
    "geo_status": "pending",   # pending | ok | denied | unavailable
    "manual_market": None,     # chosen in settings or on the crop page
}
OTHER_DEFAULTS = {
    "step": "welcome",
    "crop": "Maize",
    "quantity_kg": 500,
    "horizon": cfg.MAX_FORECAST_HORIZON,
    # {crop: {"storage": UGX/kg/month, "loss_pct": % per month}}; crops without
    # an entry use the defaults in src/config.py
    "cost_overrides": {},
    "transport_rate": float(cfg.TRANSPORT_UGX_PER_KG_KM),
    "accepted_advice": False,
}


def init_state() -> None:
    for k, v in {**PERSONAL_DEFAULTS, **OTHER_DEFAULTS}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sign_out() -> None:
    """Forget everything personal: phone, name, location and market choice."""
    for k, v in {**PERSONAL_DEFAULTS, **OTHER_DEFAULTS}.items():
        st.session_state[k] = v
    for k in list(st.session_state.keys()):
        if k.startswith(("storage_", "loss_", "geo_component", "chat", "flash", "role_choice")):
            del st.session_state[k]


def go_to(step: str) -> None:
    st.session_state.step = step
    st.rerun()


# -----------------------------------------------------------------------------
# Cached data
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading market prices…")
def load_monthly() -> pd.DataFrame:
    if not cfg.MONTHLY_PARQUET.exists():
        try:
            pipeline.build_all(write=True)
        except FileNotFoundError as e:
            st.error(str(e))
            return pd.DataFrame()
    return pipeline.load_monthly()


@st.cache_data
def load_series_summary() -> pd.DataFrame:
    if not cfg.SERIES_SUMMARY_CSV.exists():
        pipeline.build_all(write=True)
    return pipeline.load_series_summary()


@st.cache_data
def load_markets() -> pd.DataFrame:
    if not cfg.MARKETS_CSV.exists():
        pipeline.build_all(write=True)
    return pipeline.load_markets()


@st.cache_data
def load_registry() -> dict:
    return forecasting.load_registry()


@st.cache_data(show_spinner="Forecasting prices…")
def get_forecast(crop: str, market: str, horizon: int) -> forecasting.Forecast:
    return forecasting.forecast(load_monthly(), crop, market, horizon, load_registry())


@st.cache_data(show_spinner="Comparing markets…")
def get_market_comparison(crop: str, market: str, qty: float, horizon: int,
                          storage: float, loss: float, rate: float) -> decision.MarketComparison:
    return decision.compare_markets(load_monthly(), load_markets(), crop, market, qty, horizon,
                                    decision.Costs(storage, loss, rate), load_registry())


# -----------------------------------------------------------------------------
# Language model (optional)
# -----------------------------------------------------------------------------
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"


def secrets_problem() -> str | None:
    """
    Why .streamlit/secrets.toml cannot be used, or None if it is fine or absent.
    A single broken line makes Streamlit ignore the whole file, so say which line.
    """
    if not SECRETS_FILE.exists():
        return None
    import tomllib
    try:
        tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8-sig"))
        return None
    except tomllib.TOMLDecodeError as e:
        return (f"{SECRETS_FILE.relative_to(ROOT)} could not be read ({e}). Comment lines must "
                "start with #, and values need quotes, for example ADMIN_PASSWORD = \"...\". "
                "Fix the file and restart the app.")
    except (OSError, UnicodeDecodeError) as e:
        return f"{SECRETS_FILE.relative_to(ROOT)} could not be opened ({e})."


def llm_settings() -> dict:
    """Streamlit secrets, if a secrets file exists; environment variables otherwise."""
    try:
        return {k: str(v) for k, v in st.secrets.items() if isinstance(v, (str, int, float))}
    except Exception:
        return {}


def get_llm():
    return assistant.get_llm(llm_settings())


@st.cache_data(show_spinner=False, ttl=3600)
def cached_explanation(payload_json: str, model_key: str) -> assistant.AssistantReply:
    # model_key is part of the cache key so changing model re-generates
    return assistant.explain(get_llm(), json.loads(payload_json),
                             known_markets=load_markets()["market"].tolist())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def current_costs(crop: str) -> decision.Costs:
    """Crop defaults from config, replaced by anything the farmer has set."""
    c = decision.Costs.default_for(crop)
    o = st.session_state.cost_overrides.get(crop, {})
    if "storage" in o:
        c.storage_ugx_per_kg_month = float(o["storage"])
    if "loss_pct" in o:
        c.loss_rate_per_month = float(o["loss_pct"]) / 100
    c.transport_ugx_per_kg_km = float(st.session_state.transport_rate)
    return c


def pct(p: float | None) -> str:
    return "n/a" if p is None or not np.isfinite(p) else f"{p * 100:.0f}%"


def current_market(crop: str) -> flow.MarketChoice:
    return flow.resolve_market(crop, load_series_summary(), load_markets(),
                               st.session_state.coords, st.session_state.manual_market)


def detect_location() -> None:
    """Ask the browser for the farmer's position once and remember it."""
    if not GEO_AVAILABLE:
        st.session_state.geo_status = "unavailable"
        return
    if st.session_state.geo_status != "pending":
        return
    loc = get_geolocation(component_key="geo_component")
    if not loc:
        return                                   # still waiting for the browser
    if "coords" in loc and loc["coords"].get("latitude") is not None:
        st.session_state.coords = (float(loc["coords"]["latitude"]),
                                   float(loc["coords"]["longitude"]))
        st.session_state.geo_status = "ok"
    else:
        st.session_state.geo_status = "denied"


def build_explanation(adv: decision.Advice, crop: str, market: str, quantity: float,
                      fc: forecasting.Forecast, costs: decision.Costs,
                      alt: pd.Series | None) -> str:
    """Template explanation built only from computed numbers (the LLM replaces it later)."""
    parts = []
    if adv.months == 0:
        parts.append(
            f"Selling your {quantity:,.0f} kg of **{crop}** at **{market}** now, at "
            f"{fc.origin_price:,.0f} UGX/kg, gives about **{adv.net_sell_now_ugx:,.0f} UGX**."
        )
        if adv.best_store_months:
            opt = adv.options.set_index("months").loc[adv.best_store_months]
            parts.append(
                f"Storing only pays if the price rises enough to cover storage "
                f"({costs.storage_ugx_per_kg_month:,.0f} UGX/kg a month) and losses "
                f"({costs.loss_rate_per_month * 100:.2f}% a month). The best storage option, "
                f"{opt['action'].lower()}, needs at least **{opt['breakeven_price']:,.0f} UGX/kg**, "
                f"and the chance of reaching that is about **{pct(adv.best_store_prob)}**."
            )
    else:
        parts.append(
            f"**{adv.recommendation}** is expected to earn about **{adv.gain_ugx:,.0f} UGX** "
            f"more than selling your {quantity:,.0f} kg of {crop} at {market} now, after storage "
            f"costs and losses. The chance it beats selling now is about **{pct(adv.confidence)}**."
        )
    parts.append(f"Confidence: **{adv.confidence_label}** ({pct(adv.confidence)}).")
    if fc.model == "naive":
        parts.append(
            "The forecast assumes the price stays at its latest recorded level, because on "
            "past data no method has yet predicted these prices better than that. The "
            "chances above come from how far real prices strayed from such forecasts before."
        )
    else:
        parts.append(f"Forecast method: {fc.label}, chosen because it beat the simple baseline on past data.")
    if alt is not None and alt["best_net_ugx"] > adv.expected_net_ugx:
        parts.append(
            f"**{alt['market']}** ({alt['road_km']:,.0f} km away) could pay about "
            f"{alt['best_net_ugx'] - adv.expected_net_ugx:,.0f} UGX more even after "
            f"{alt['transport_ugx']:,.0f} UGX of transport. See the market comparison."
        )
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Sidebar: account and settings (no step buttons; the flow drives navigation)
# -----------------------------------------------------------------------------
def sidebar() -> None:
    st.sidebar.title("🌾 AgriPrice Advisor")
    if not st.session_state.logged_in:
        st.sidebar.info("Not signed in")
        return

    who = st.session_state.farmer_name or st.session_state.phone or st.session_state.email
    role = st.session_state.role
    st.sidebar.success(f"Signed in · {who}" + ("" if role == "farmer" else f"  \n{ROLE_LABELS[role]}"))
    if role != "farmer":
        if st.sidebar.button("Sign out"):
            sign_out()
            st.rerun()
        return

    if st.sidebar.button("My advice", width="stretch"):
        go_to("dashboard" if st.session_state.advice_ctx else "crop")
    waiting = services_ui.get_store().farmer_bookings(
        st.session_state.phone or st.session_state.email)
    n_quotes = int((waiting["status"] == "quoted").sum()) if not waiting.empty else 0
    label = f"My bookings ({n_quotes} quotation{'s' if n_quotes != 1 else ''} waiting)" \
        if n_quotes else "My bookings"
    if st.sidebar.button(label, width="stretch"):
        go_to("bookings")

    with st.sidebar.expander("Settings"):
        crop = st.session_state.crop
        choice = current_market(crop)
        options = flow.available_markets(load_series_summary(), crop)
        labels = [AUTO] + options
        current = st.session_state.manual_market
        idx = labels.index(current) if current in labels else 0
        pick = st.selectbox(f"Your market for {crop}", labels, index=idx,
                            help="Automatic uses the nearest market with recent prices "
                                 "for this crop.")
        new_manual = None if pick == AUTO else pick
        if new_manual != st.session_state.manual_market:
            st.session_state.manual_market = new_manual
            st.rerun()
        if choice.market and choice.source == "location":
            st.caption(f"Nearest market with recent {crop} prices: {choice.market} "
                       f"(about {choice.distance_km:,.0f} km).")
        status = {
            "ok": "Location detected.",
            "pending": "Waiting for your browser to share your location.",
            "denied": "Location not shared, so choose your market above.",
            "unavailable": "Location detection is not installed (pip install streamlit-js-eval).",
        }[st.session_state.geo_status]
        st.caption(status)
        if st.session_state.geo_status in ("ok", "denied") and st.button("Detect my location again"):
            st.session_state.coords = None
            st.session_state.geo_status = "pending"
            st.session_state.pop("geo_component", None)
            st.rerun()

    if st.sidebar.button("Sign out"):
        sign_out()
        st.rerun()


# -----------------------------------------------------------------------------
# Step 1: sign in
# -----------------------------------------------------------------------------
def page_welcome() -> None:
    st.header("Welcome to AgriPrice Advisor")
    st.markdown(
        "Price forecasts and **sell-or-store advice** for smallholder farmers in Uganda. "
        "Each crop is forecast with whichever method has been most accurate on past "
        "prices, and every recommendation is explained."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sign in or register")
        role = st.radio("I am a", list(ROLE_LABELS), format_func=ROLE_LABELS.get,
                        horizontal=True, key="role_choice")
        if role == "admin":
            with st.form("admin_sign_in"):
                password = st.text_input("Administrator password", type="password")
                ok = st.form_submit_button("Sign in", type="primary")
            if ok:
                expected = llm_settings().get("ADMIN_PASSWORD") or os.environ.get("ADMIN_PASSWORD")
                problem = secrets_problem()
                if problem:
                    st.error(problem)
                elif not expected:
                    st.error("Administrator sign-in is not set up. Add ADMIN_PASSWORD to "
                             ".streamlit/secrets.toml and restart the app.")
                elif not hmac.compare_digest(password, expected):
                    st.error("Wrong password.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.role = "admin"
                    st.session_state.farmer_name = "Administrator"
                    go_to("admin")
            return
        with st.form("sign_in"):
            contact_raw = st.text_input("Phone number or email",
                                        placeholder="e.g. 0772 123 456 or name@example.com")
            name = st.text_input("Your name (optional)")
            consent_text = (
                "I agree that my contact, location and crop details are used only to give me "
                "price advice and arrange the services I book." if role == "farmer" else
                "I agree that my business details are checked by an administrator and shown "
                "to farmers, and that my contact is shared with farmers whose bookings I accept.")
            consent = st.checkbox(consent_text)
            submitted = st.form_submit_button("Continue", type="primary")
        if submitted:
            contact = flow.parse_contact(contact_raw)
            if contact is None:
                if "@" in (contact_raw or ""):
                    st.error("That email address doesn't look right. Check it, for "
                             "example name@example.com.")
                else:
                    st.error("Please enter a Ugandan mobile number (for example 0772 123 456 "
                             "or +256 772 123 456) or an email address.")
            elif not consent:
                st.error("Please tick the consent box to continue.")
            else:
                kind, value = contact
                st.session_state.logged_in = True
                st.session_state.phone = value if kind == "phone" else ""
                st.session_state.email = value if kind == "email" else ""
                st.session_state.farmer_name = name.strip()
                st.session_state.role = role
                go_to(ROLE_HOME[role])
    with col2:
        st.info(
            "**How it works**\n\n"
            "1. Tell us your crop and how much you have\n"
            "2. We find your nearest market from your location\n"
            "3. You get a clear sell-or-store recommendation, with the reasons\n\n"
            "If you accept the advice you can book storage, transport or a buyer from "
            "verified providers.\n\n"
            "**Service providers and buyers** can register here and receive requests."
        )


# -----------------------------------------------------------------------------
# Step 2: crop and quantity (location found in the background)
# -----------------------------------------------------------------------------
def page_crop() -> None:
    st.header("What are you selling?")
    detect_location()
    summary = load_series_summary()

    crops = flow.available_crops(summary)
    if not crops:
        st.error("No crop has recent prices. Rebuild the data with scripts/build_dataset.py.")
        return
    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", crops,
                        index=crops.index(st.session_state.crop) if st.session_state.crop in crops else 0)
    quantity = c2.number_input("Quantity (kg)", min_value=50, max_value=50_000,
                               value=int(st.session_state.quantity_kg), step=50)
    st.session_state.crop = crop
    st.session_state.quantity_kg = quantity

    choice = current_market(crop)
    if choice.market is None:
        options = flow.available_markets(summary, crop)
        msg = {
            "pending": "Finding your location… If your browser asks, please allow it. "
                       "Or choose your market now:",
            "denied": "We could not get your location. Please choose your nearest market:",
            "unavailable": "Please choose your nearest market:",
            "ok": "Please choose your nearest market:",
        }[st.session_state.geo_status]
        if choice.note:
            st.caption(choice.note)
        pick = st.selectbox(msg, options, index=None, placeholder="Choose a market")
        if pick:
            st.session_state.manual_market = pick
            st.rerun()
    else:
        if choice.source == "location":
            where = (" (less than 1 km from you)" if choice.distance_km < 1
                     else f" (about {choice.distance_km:,.0f} km from you)")
        else:
            where = ""
        st.caption(f"Market: **{choice.market}**{where}. Change it in Settings.")
        if choice.note:
            st.caption(choice.note)

    costs = current_costs(crop)
    with st.expander("Costs and forecast length (optional)"):
        st.caption(f"Defaults for {crop} come from published figures where they exist. "
                   "Replace them with what you actually pay.")
        storage = st.slider("Storage cost (UGX per kg per month)", 0, 100,
                            int(round(costs.storage_ugx_per_kg_month)), key=f"storage_{crop}")
        loss = st.slider("Storage loss (% per month)", 0.0, 5.0,
                         round(costs.loss_rate_per_month * 100, 2), 0.05, key=f"loss_{crop}")
        rate = st.number_input("Transport to other markets (UGX per kg per km)", 0.0, 50.0,
                               float(st.session_state.transport_rate), step=0.5)
        max_h = cfg.MAX_FORECAST_HORIZON
        horizon = st.slider("Look ahead (months)", 1, max_h,
                            min(int(st.session_state.horizon), max_h))
    st.session_state.cost_overrides[crop] = {"storage": storage, "loss_pct": loss}
    st.session_state.transport_rate = rate
    st.session_state.horizon = horizon

    if st.button("Get my advice →", type="primary", disabled=choice.market is None):
        go_to("dashboard")


# -----------------------------------------------------------------------------
# Step 3: advice dashboard
# -----------------------------------------------------------------------------
def price_chart(series: pd.DataFrame, fc: forecasting.Forecast, crop: str, market: str) -> bool:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["date"], y=series["price"], mode="lines",
                             name="Past prices", line=dict(color="#2e7d32", width=2)))
    x_fc = [fc.origin_date] + list(fc.dates)
    has_band = all(np.isfinite(fc.lower)) and all(np.isfinite(fc.upper))
    if has_band:
        fig.add_trace(go.Scatter(
            x=x_fc + x_fc[::-1],
            y=[fc.origin_price] + fc.upper + (fc.lower[::-1] + [fc.origin_price]),
            mode="lines", fill="toself", fillcolor="rgba(249,168,37,0.18)", line=dict(width=0),
            hoverinfo="skip", name=f"Likely range ({fc.level:.0%} of past outcomes)"))
    fig.add_trace(go.Scatter(x=x_fc, y=[fc.origin_price] + list(fc.median),
                             mode="lines+markers", name="Forecast",
                             line=dict(color="#f9a825", dash="dash")))
    start = fc.origin_date - pd.DateOffset(years=cfg.CHART_DEFAULT_YEARS)
    end = fc.dates[-1] + pd.DateOffset(months=1)
    recent = series[series["date"] >= start]
    lo = np.nanmin([recent["price"].min(), min(fc.lower) if has_band else np.inf])
    hi = np.nanmax([recent["price"].max(), max(fc.upper) if has_band else -np.inf])
    fig.update_layout(
        xaxis_title="", yaxis_title=f"{crop} at {market}, UGX per kg", height=380,
        hovermode="x unified", margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(range=[start, end])
    fig.update_yaxes(range=[lo * 0.9, hi * 1.08])
    st.plotly_chart(fig, width="stretch")
    return has_band


def market_comparison_view(mc: decision.MarketComparison, adv: decision.Advice,
                           alt: pd.Series | None, crop: str, market: str) -> None:
    t = mc.table
    st.caption(f"Compared using {crop} prices recorded in {mc.reference_month:%B %Y}, "
               "the same month as your market's latest price.")
    if alt is None:
        st.write(f"No other market has a {crop} price from {mc.reference_month:%B %Y}, "
                 "so there is nothing to compare fairly yet.")
    else:
        home = t[t["is_home"]].iloc[0]
        diff = alt["best_net_ugx"] - adv.expected_net_ugx
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Your market: {market}**")
            st.metric("Price", f"{home['latest_price']:,.0f} UGX/kg")
            st.metric("Best option there", adv.recommendation)
            st.metric("Earnings", f"{adv.expected_net_ugx:,.0f} UGX")
        with b:
            st.markdown(f"**Best alternative: {alt['market']}** ({alt['road_km']:,.0f} km)")
            st.metric("Price", f"{alt['latest_price']:,.0f} UGX/kg")
            st.metric("Best option there", alt["best_option"],
                      help=f"Includes {alt['transport_ugx']:,.0f} UGX transport")
            st.metric("Earnings after transport", f"{alt['best_net_ugx']:,.0f} UGX",
                      f"{diff:+,.0f} UGX vs your market")
        if diff > 0:
            st.success(f"Selling at **{alt['market']}** could earn about {diff:,.0f} UGX more "
                       f"after {alt['transport_ugx']:,.0f} UGX of transport. Distances are "
                       "estimates, so check the real route and cost.")
        else:
            st.info(f"**{market}** gives the best earnings once transport is counted.")

        with st.expander(f"See all {len(t)} compared markets"):
            st.dataframe(pd.DataFrame({
                "Market": t["market"] + np.where(t["is_home"], " (yours)", ""),
                "Region": t["region"],
                "Road distance (km)": t["road_km"].round(0),
                "Latest price (UGX/kg)": t["latest_price"].round(0),
                "Transport (UGX)": t["transport_ugx"].round(0),
                "Best option there": t["best_option"],
                "Best earnings (UGX)": t["best_net_ugx"].round(0),
                "vs selling now at your market": t["vs_home_sell_now_ugx"].round(0),
                "Note": np.where(t["sudden_jump"] & ~t["is_home"],
                                 "Price jumped suddenly, may be an error", ""),
            }), width="stretch", hide_index=True)
            if (t["sudden_jump"] & ~t["is_home"]).any():
                st.caption("Markets whose latest price jumped by about half or more since the "
                           "month before are not suggested, because the jump may be a "
                           "recording error. Check the price before travelling.")
    if mc.excluded:
        st.caption(f"{len(mc.excluded)} market(s) left out because their latest {crop} "
                   "price is from a different month.")


def explanation_section(payload: dict, adv, crop, market, qty, fc, costs, alt) -> None:
    st.subheader("Why this advice?")
    llm = get_llm()
    template = build_explanation(adv, crop, market, qty, fc, costs, alt)
    if llm is None:
        st.markdown(template)
        return
    payload_json = json.dumps(payload, sort_keys=True)
    with st.spinner("Writing your explanation…"):
        reply = cached_explanation(payload_json, f"{llm.name}:{llm.model}")
    if reply.ok:
        st.markdown(reply.text)
        st.caption("Written by AI from the figures on this page. Every number in it was checked "
                   "against the system's results.")
        with st.expander("Show the standard explanation"):
            st.markdown(template)
    else:
        st.markdown(template)
        st.caption(f"The AI explanation is not available ({reply.reason}), so the standard "
                   "explanation is shown.")


SUGGESTED_QUESTIONS = [
    "Why should I sell now and not store?",
    "How sure is this advice?",
    "Is another market better for me?",
]


def question_section(payload: dict) -> None:
    st.subheader("Ask about this advice")
    llm = get_llm()
    if llm is None:
        st.caption("The question assistant needs a language model. Add an API key in "
                   ".streamlit/secrets.toml (see secrets.toml.example) to turn it on.")
        return

    # Keep the conversation only while the advice it is about stays the same.
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if st.session_state.get("chat_key") != key:
        st.session_state.chat_key = key
        st.session_state.chat = []
    chat = st.session_state.chat

    suggested = list(SUGGESTED_QUESTIONS)
    if payload["recommendation"]["action"] != "Sell now":
        suggested[0] = "Why should I store instead of selling now?"
    cols = st.columns(len(suggested))
    clicked = None
    for c, q in zip(cols, suggested):
        if c.button(q, key=f"suggest_{q}"):
            clicked = q
    with st.form("ask", clear_on_submit=True):
        typed = st.text_input("Your question", placeholder="e.g. What happens if prices fall?")
        sent = st.form_submit_button("Ask")
    question = clicked or (typed if sent else None)

    if question:
        with st.spinner("Thinking…"):
            reply = assistant.answer(llm, payload, question, chat,
                                     known_markets=load_markets()["market"].tolist())
        text = reply.text if reply.text else (
            "Sorry, I could not answer that just now. Please try again in a moment.")
        chat.append({"role": "user", "content": question})
        chat.append({"role": "assistant", "content": text})

    for turn in chat:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])
    if chat:
        st.caption("Answers use only the figures on this page. The assistant cannot see "
                   "weather, buyers or service providers.")


def page_dashboard() -> None:
    crop = st.session_state.crop
    qty = st.session_state.quantity_kg
    horizon = min(int(st.session_state.horizon), cfg.MAX_FORECAST_HORIZON)
    choice = current_market(crop)
    if choice.market is None:
        go_to("crop")
    market = choice.market

    series = pipeline.get_series(load_monthly(), crop, market)
    series = series[series["price"].notna()]
    if series.empty:
        st.warning(f"No {crop} price history for {market}.")
        if st.button("← Choose another crop or market"):
            go_to("crop")
        return

    fc = get_forecast(crop, market, horizon)
    costs = current_costs(crop)
    adv = decision.advise(fc, qty, costs)
    mc = get_market_comparison(crop, market, qty, horizon, costs.storage_ugx_per_kg_month,
                               costs.loss_rate_per_month, costs.transport_ugx_per_kg_km)
    alt = decision.best_alternative(mc)

    st.header(f"Your advice: {qty:,.0f} kg of {crop} at {market}")

    summary = load_series_summary()
    row = summary[(summary["commodity"] == crop) & (summary["market"] == market)]
    if len(row) and bool(row["is_stale"].iloc[0]):
        st.warning(f"The latest {crop} price for {market} is from {fc.origin_date:%B %Y}, "
                   f"{int(row['months_since_last_obs'].iloc[0])} months older than the newest "
                   "data in the system. Treat this advice with caution.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest price", f"{fc.origin_price:,.0f} UGX/kg", help=f"Recorded {fc.origin_date:%B %Y}")
    c1.caption(f"Recorded {fc.origin_date:%B %Y}")
    c2.metric("Recommendation", adv.recommendation)
    if adv.months == 0:
        c3.metric("Earnings if you sell now", f"{adv.net_sell_now_ugx:,.0f} UGX")
    else:
        c3.metric("Extra earnings vs selling now", f"{adv.gain_ugx:,.0f} UGX")
    c4.metric("Confidence", pct(adv.confidence),
              help="Chance that the recommended choice is the better one, from the size of "
                   "past forecast errors")
    c4.caption(adv.confidence_label)

    if adv.months > 0:
        st.success(f"**{adv.recommendation}** looks better than selling now.")
    elif adv.best_store_prob is not None:
        st.info(f"**Sell now.** Storing has about a {pct(adv.best_store_prob)} chance of paying "
                f"off at best, below the {cfg.STORE_MIN_PROBABILITY:.0%} needed to advise it.")
    else:
        st.info("**Sell now** is recommended under the current assumptions.")
    if alt is not None and alt["best_net_ugx"] > adv.expected_net_ugx:
        st.caption(f"Tip: **{alt['market']}** may pay about "
                   f"{alt['best_net_ugx'] - adv.expected_net_ugx:,.0f} UGX more after transport. "
                   "See *Other markets*.")

    tab1, tab2, tab3 = st.tabs(["Price forecast", "Sell or store options", "Other markets"])
    with tab1:
        has_band = price_chart(series, fc, crop, market)
        errs = [e for e in fc.typical_error_pct if e is not None]
        cap = f"Forecast method: **{fc.label}**."
        if errs:
            cap += (f" On past data its typical error was about ±{errs[0]:.0f}% one month "
                    f"ahead and ±{errs[-1]:.0f}% at {len(errs)} months.")
        if has_band:
            cap += (f" The shaded range held the real price in {fc.level:.0%} of past "
                    "forecasts; real prices can still fall outside it.")
        cap += f" The chart shows the last {cfg.CHART_DEFAULT_YEARS} years; zoom out to see more."
        st.caption(cap)
        if fc.note:
            st.caption(fc.note)
        with st.expander("Forecast table"):
            st.dataframe(
                fc.to_frame().assign(date=lambda d: d["date"].dt.strftime("%b %Y")).rename(columns={
                    "date": "Month", "forecast": "Forecast (UGX/kg)", "lower": "Low end",
                    "upper": "High end", "months_ahead": "Months ahead",
                    "typical_error_pct": "Typical error (%)"}),
                width="stretch", hide_index=True)

    with tab2:
        o = adv.options
        st.dataframe(pd.DataFrame({
            "Option": o["action"],
            "Price (UGX/kg)": o["price"].round(0),
            "Likely range": [("" if m == 0 else f"{lo:,.0f} to {hi:,.0f}")
                             for m, lo, hi in zip(o["months"], o["low"], o["high"])],
            "Kg left to sell": o["kg_sold"].round(0),
            "Storage cost (UGX)": o["storage_ugx"].round(0),
            "Expected earnings (UGX)": o["net_ugx"].round(0),
            "vs selling now (UGX)": o["gain_vs_now_ugx"].round(0),
            "Break-even price": o["breakeven_price"].round(0),
            "Chance it beats selling now": [pct(p) if m else "" for m, p in
                                            zip(o["months"], o["prob_beats_now"])],
        }), width="stretch", hide_index=True)
        st.caption(f"Costs used: storage {costs.storage_ugx_per_kg_month:,.0f} UGX/kg per month, "
                   f"loss {costs.loss_rate_per_month * 100:.2f}% per month. Break-even price is "
                   "the price at which storing earns the same as selling now.")

    with tab3:
        market_comparison_view(mc, adv, alt, crop, market)

    months_old = int(row["months_since_last_obs"].iloc[0]) if len(row) else None
    payload = assistant.build_payload(
        crop=crop, market=market, quantity_kg=qty, fc=fc, adv=adv, costs=costs, mc=mc, alt=alt,
        months_old=months_old, is_stale=bool(len(row) and row["is_stale"].iloc[0]))
    explanation_section(payload, adv, crop, market, qty, fc, costs, alt)
    question_section(payload)

    st.markdown("---")
    a, b = st.columns(2)
    if a.button("Follow this advice: arrange services →", type="primary"):
        go_alt = alt is not None and alt["best_net_ugx"] > adv.expected_net_ugx
        st.session_state.accepted_advice = True
        st.session_state.chosen_provider = None
        st.session_state.advice_ctx = {
            "crop": crop, "quantity_kg": float(qty), "market": market,
            "recommendation": adv.recommendation, "months": adv.months,
            "sell_market": alt["market"] if go_alt else market,
            "latest_price": fc.origin_price, "confidence": adv.confidence,
            "transport_rate": costs.transport_ugx_per_kg_km,
        }
        go_to("services")
    if b.button("← Change crop or quantity"):
        go_to("crop")


# -----------------------------------------------------------------------------
# Part 2: services (screens in app/services_ui.py)
# -----------------------------------------------------------------------------
def page_services() -> None:
    services_ui.farmer_services_page(load_markets())


def page_bookings() -> None:
    services_ui.farmer_bookings_page(load_markets(), load_monthly, load_registry)


def page_provider() -> None:
    services_ui.provider_portal(load_markets())


def page_admin() -> None:
    services_ui.admin_page()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    init_state()
    if load_monthly().empty:
        st.error(f"No price data could be loaded. Run scripts/build_dataset.py in `{ROOT}`.")
        st.stop()

    # Every page except sign-in needs a signed-in user, and each role only
    # reaches its own pages.
    if st.session_state.step not in STEPS:
        st.session_state.step = "welcome"
    if not st.session_state.logged_in:
        st.session_state.step = "welcome"
    elif st.session_state.step not in ROLE_STEPS[st.session_state.role]:
        st.session_state.step = ROLE_HOME[st.session_state.role]

    sidebar()
    {"welcome": page_welcome, "crop": page_crop, "dashboard": page_dashboard,
     "services": page_services, "bookings": page_bookings,
     "provider": page_provider, "admin": page_admin}[st.session_state.step]()

    st.markdown("---")
    st.caption("AgriPrice Advisor · Group 6 · Data: WFP Uganda market prices")


if __name__ == "__main__":
    main()
