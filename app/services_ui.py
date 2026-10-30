"""
Streamlit screens for Part 2 of the user flow (Phase 9).

    farmer_services_page()   ranked options after accepting the advice, and booking
    farmer_bookings_page()   quotations, confirmed bookings, updates, ratings
    provider_portal()        register, respond to requests, post updates
    admin_page()             verify providers, overview

All rules live in src/services.py; this file only draws the screens.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from src import config as cfg
from src import decision, forecasting
from src.services import STATUS_LABELS, ServiceError, ServiceStore, maps_link


@st.cache_resource
def get_store() -> ServiceStore:
    return ServiceStore(cfg.SERVICES_DB)


def _kind_label(kind: str) -> str:
    return cfg.SERVICE_KINDS[kind]["label"]


def _price_text(kind: str, price: float) -> str:
    return f"{price:,.0f} {cfg.SERVICE_KINDS[kind]['price_unit']}"


def _stars(avg, n) -> str:
    if avg is None or (isinstance(avg, float) and np.isnan(avg)) or not n:
        return "No ratings yet"
    return f"{'★' * int(round(avg))}{'☆' * (5 - int(round(avg)))} {avg:.1f} ({int(n)})"


def _show_error(e: Exception) -> None:
    st.error(str(e))


# =============================================================================
# Farmer: ranked options and booking
# =============================================================================
def _options_for_advice(ctx: dict) -> list[tuple[str, str, str]]:
    """(kind, heading, where) groups to offer, recommended first."""
    if ctx["months"] > 0:
        return [("storage", f"Storage near {ctx['market']} for {ctx['months']} month(s)", ctx["market"]),
                ("preservation", "Drying and treatment before storage", ctx["market"])]
    groups = []
    if ctx.get("sell_market") and ctx["sell_market"] != ctx["market"]:
        groups.append(("transport", f"Transport from {ctx['market']} to {ctx['sell_market']}",
                       ctx["market"]))
    groups.append(("buyer", f"Buyers for your {ctx['crop'].lower()}", ctx["market"]))
    return groups


def _option_card(store: ServiceStore, row: pd.Series, ctx: dict, kind: str, i: int) -> None:
    with st.container(border=True):
        a, b, c = st.columns([3, 2, 2])
        where = ("at your market" if row["distance_km"] < 1
                 else f"about {row['distance_km']:,.0f} km away")
        a.markdown(f"**{row['name']}**  \n{row['market']} · {where}")
        a.caption(row["description"] or _kind_label(kind))
        b.markdown(f"{_price_text(kind, row['price'])}  \n"
                   + (f"Up to {row['capacity_kg']:,.0f} kg per trip"
                      if kind == "transport" else f"Up to {row['capacity_kg']:,.0f} kg"))
        b.caption(_stars(row["rating"], row["rating_count"]))
        if kind == "buyer":
            c.metric("Estimated earnings", f"{row['estimate_ugx']:,.0f} UGX",
                     help="Price paid × your kg, minus transport to the buyer")
        else:
            c.metric("Estimated cost", f"{row['estimate_ugx']:,.0f} UGX",
                     help="From the provider's listed price; the quotation may differ")
        if st.button("Choose this option", key=f"choose_{kind}_{row['id']}_{i}"):
            st.session_state.chosen_provider = int(row["id"])
            st.rerun()


def _booking_form(store: ServiceStore, ctx: dict, provider_id: int) -> None:
    p = store.provider(provider_id)
    kind = p["kind"]
    st.subheader(f"Request: {p['name']}")
    st.caption(f"{_kind_label(kind)} · {p['market']} · {_price_text(kind, p['price'])}")
    with st.form("booking_form"):
        qty = st.number_input("Quantity (kg)", min_value=50, max_value=100_000,
                              value=int(ctx["quantity_kg"]), step=50)
        today = date.today()
        if kind == "storage":
            start = st.date_input("Drop-off date", today + timedelta(days=3), min_value=today)
            end = st.date_input("Planned collection and sale date",
                                start + timedelta(days=30 * max(ctx["months"], 1)), min_value=today)
        else:
            start = st.date_input("Date", today + timedelta(days=3), min_value=today)
            end = None
        dest, share = None, False
        if kind == "transport":
            dest = ctx.get("sell_market") or ctx["market"]
            st.markdown(f"From **{ctx['market']}** to **{dest}**")
            share = st.checkbox("Share the trip with other farmers going the same way "
                                "(cost split by kg)", value=True)
        submitted = st.form_submit_button("Send request", type="primary")
    if kind == "transport":
        est = store.sharing_estimate(provider_id, ctx["market"], dest, start, qty)
        if est["others"]:
            st.info(f"{est['others']} other farmer(s) with {est['others_kg']:,.0f} kg have asked for this "
                    f"route around that date. Sharing would cost you about {est['shared_ugx']:,.0f} UGX "
                    f"instead of {est['alone_ugx']:,.0f} UGX.")
    if submitted:
        try:
            store.request_booking(
                farmer_contact=ctx["contact"], provider_id=provider_id, crop=ctx["crop"],
                quantity_kg=qty, start_date=start, end_date=end, origin_market=ctx["market"],
                destination_market=dest, share_transport=share,
                advice_snapshot={k: ctx[k] for k in ("recommendation", "months", "sell_market",
                                                     "latest_price", "confidence") if k in ctx})
        except ServiceError as e:
            _show_error(e)
            return
        st.session_state.chosen_provider = None
        st.session_state.step = "bookings"
        st.session_state.flash = (f"Request sent to {p['name']}. They have "
                                  f"{cfg.QUOTE_DEADLINE_HOURS} hours to reply with a quotation.")
        st.rerun()
    if st.button("← Back to the options"):
        st.session_state.chosen_provider = None
        st.rerun()


def farmer_services_page(markets: pd.DataFrame) -> None:
    store = get_store()
    ctx = st.session_state.get("advice_ctx")
    if not ctx:
        st.info("Get your advice first, then choose services to carry it out.")
        if st.button("Get advice"):
            st.session_state.step = "crop"
            st.rerun()
        return
    ctx = {**ctx, "contact": st.session_state.phone or st.session_state.email}

    if st.session_state.get("chosen_provider"):
        _booking_form(store, ctx, st.session_state.chosen_provider)
        return

    st.header("Carry out your plan")
    st.markdown(f"Your advice: **{ctx['recommendation']}** for {ctx['quantity_kg']:,.0f} kg of "
                f"{ctx['crop']} at {ctx['market']}"
                + (f", selling at **{ctx['sell_market']}**" if ctx.get("sell_market") != ctx["market"] else "")
                + ". Options from verified providers, best first:")

    shown_any = False
    for kind, heading, where in _options_for_advice(ctx):
        ranked = store.rank_options(kind=kind, crop=ctx["crop"], quantity_kg=ctx["quantity_kg"],
                                    home_market=where, markets=markets, months=ctx["months"],
                                    transport_ugx_per_kg_km=ctx["transport_rate"])
        st.subheader(heading)
        if ranked.empty:
            st.caption("No verified provider offers this yet.")
            continue
        shown_any = True
        n = cfg.RANKED_OPTIONS_SHOWN
        for i, (_, row) in enumerate(ranked.head(n).iterrows()):
            _option_card(store, row, ctx, kind, i)
        if len(ranked) > n:
            with st.expander(f"Show {len(ranked) - n} more"):
                for i, (_, row) in enumerate(ranked.iloc[n:].iterrows(), start=n):
                    _option_card(store, row, ctx, kind, i)

    with st.expander("Other services"):
        others = [k for k in cfg.SERVICE_KINDS if k not in {g[0] for g in _options_for_advice(ctx)}]
        kind = st.selectbox("Service", others, format_func=_kind_label)
        ranked = store.rank_options(kind=kind, crop=ctx["crop"], quantity_kg=ctx["quantity_kg"],
                                    home_market=ctx["market"], markets=markets,
                                    months=max(ctx["months"], 1),
                                    transport_ugx_per_kg_km=ctx["transport_rate"])
        if ranked.empty:
            st.caption("No verified provider offers this yet.")
        for i, (_, row) in enumerate(ranked.head(5).iterrows()):
            _option_card(store, row, {**ctx, "sell_market": ctx.get("sell_market")}, kind, 100 + i)

    if not shown_any:
        st.info("No verified providers match your plan yet. Try again later, or look under "
                "Other services.")
    c1, c2 = st.columns(2)
    if c1.button("← Back to my advice"):
        st.session_state.step = "dashboard"
        st.rerun()
    if c2.button("My bookings"):
        st.session_state.step = "bookings"
        st.rerun()


# =============================================================================
# Farmer: my bookings
# =============================================================================
def _timeline(store: ServiceStore, booking_id: int) -> None:
    ev = store.events(booking_id)
    for _, e in ev.iterrows():
        when = pd.Timestamp(e["at"]).strftime("%d %b %Y %H:%M")
        what = STATUS_LABELS.get(e["status"], "Update") if isinstance(e["status"], str) else "Update"
        line = f"- {when} · **{what}**"
        if e["note"]:
            line += f": {e['note']}"
        if e["condition"]:
            line += f" (condition: {e['condition']})"
        st.markdown(line)


def _recheck_advice(b: pd.Series, markets: pd.DataFrame, load_monthly, load_registry) -> None:
    """Before the planned sale date, re-run the forecast and advice on current data."""
    try:
        fc = forecasting.forecast(load_monthly(), b["crop"], b["origin_market"],
                                  cfg.MAX_FORECAST_HORIZON, load_registry())
    except ValueError:
        return
    adv = decision.advise(fc, b["quantity_kg"], decision.Costs.default_for(b["crop"]))
    st.info(f"Advice re-checked with the latest prices ({fc.origin_date:%B %Y}): "
            f"**{adv.recommendation}** for {b['crop']} at {b['origin_market']}, "
            f"latest price {fc.origin_price:,.0f} UGX/kg, confidence {adv.confidence_label}.")


def farmer_bookings_page(markets: pd.DataFrame, load_monthly, load_registry) -> None:
    store = get_store()
    store.expire_overdue()
    contact = st.session_state.phone or st.session_state.email
    st.header("My bookings")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    df = store.farmer_bookings(contact)
    if df.empty:
        st.info("You have no bookings yet. Accept your advice to see service options.")
    mk = markets.set_index("market")
    for _, b in df.iterrows():
        status = b["status"]
        title = (f"{_kind_label(b['kind'])} · {b['provider_name']} · {b['quantity_kg']:,.0f} kg "
                 f"{b['crop']} · {STATUS_LABELS[status]}")
        with st.expander(title, expanded=status in ("quoted", "expired", "declined_by_provider")):
            dates = f"from {b['start_date']}" + (f" to {b['end_date']}" if b["end_date"] else "")
            route = (f"{b['origin_market']} → {b['destination_market']}"
                     if b["destination_market"] else b["origin_market"])
            st.caption(f"Booking {b['id']} · {route} · {dates}"
                       + (" · shared trip" if b["share_transport"] else ""))

            if status == "quoted":
                st.markdown(f"Quotation: **{b['quote_ugx']:,.0f} UGX**"
                            + (f". {b['quote_note']}" if b["quote_note"] else ""))
                c1, c2 = st.columns(2)
                if c1.button("Accept quotation", key=f"acc_{b['id']}", type="primary"):
                    try:
                        store.accept_quote(int(b["id"]), contact)
                        st.rerun()
                    except ServiceError as e:
                        _show_error(e)
                if c2.button("Decline", key=f"dec_{b['id']}"):
                    store.decline_quote(int(b["id"]), contact)
                    st.rerun()

            if status in ("accepted", "in_progress", "completed"):
                lat, lon = mk.at[b["provider_market"], "latitude"], mk.at[b["provider_market"], "longitude"]
                st.markdown(f"Agreed price: **{b['quote_ugx']:,.0f} UGX** · Contact: "
                            f"**{b['provider_contact']}** · [Map of {b['provider_market']}]"
                            f"({maps_link(lat, lon)})")

            if status in ("expired", "declined_by_provider"):
                st.warning("This provider did not take the request. Here is the next best option.")
                nxt = store.next_ranked(int(b["id"]), markets)
                if nxt.empty:
                    st.caption("No other verified provider is available right now.")
                else:
                    r = nxt.iloc[0]
                    st.markdown(f"**{r['name']}**, {r['market']} · {_price_text(r['kind'], r['price'])}")
                    if st.button(f"Send the same request to {r['name']}", key=f"next_{b['id']}"):
                        try:
                            store.request_booking(
                                farmer_contact=contact, provider_id=int(r["id"]), crop=b["crop"],
                                quantity_kg=b["quantity_kg"],
                                start_date=date.fromisoformat(b["start_date"]),
                                end_date=date.fromisoformat(b["end_date"]) if b["end_date"] else None,
                                origin_market=b["origin_market"],
                                destination_market=b["destination_market"],
                                share_transport=bool(b["share_transport"]))
                            st.session_state.flash = f"Request sent to {r['name']}."
                            st.rerun()
                        except ServiceError as e:
                            _show_error(e)

            if status in ("requested", "quoted", "accepted"):
                if st.button("Cancel booking", key=f"can_{b['id']}"):
                    store.cancel(int(b["id"]), contact)
                    st.rerun()

            if b["kind"] == "storage" and status in ("accepted", "in_progress") and b["end_date"]:
                if date.fromisoformat(b["end_date"]) - date.today() <= timedelta(days=30):
                    _recheck_advice(b, markets, load_monthly, load_registry)

            if status == "completed":
                if pd.isna(b["rating_stars"]):
                    with st.form(f"rate_{b['id']}"):
                        stars = st.slider("How was this provider?", 1, 5, 4)
                        price = st.number_input("Price you finally sold at (UGX per kg, optional)",
                                                min_value=0, value=0, step=50)
                        comment = st.text_input("Comment (optional)")
                        if st.form_submit_button("Submit rating"):
                            try:
                                store.rate(int(b["id"]), contact, stars, comment, price or None)
                                st.rerun()
                            except ServiceError as e:
                                _show_error(e)
                else:
                    st.caption(f"You rated this {int(b['rating_stars'])}/5"
                               + (f", sold at {b['price_received_ugx_per_kg']:,.0f} UGX/kg"
                                  if pd.notna(b["price_received_ugx_per_kg"]) else ""))
            st.markdown("**History**")
            _timeline(store, int(b["id"]))

    if st.button("← Back to my advice"):
        st.session_state.step = "dashboard" if st.session_state.get("advice_ctx") else "crop"
        st.rerun()


# =============================================================================
# Service provider portal
# =============================================================================
def _registration_form(store: ServiceStore, contact: str, markets: pd.DataFrame, key: str) -> None:
    with st.form(key):
        name = st.text_input("Business name")
        kind = st.selectbox("Service you offer", list(cfg.SERVICE_KINDS), format_func=_kind_label)
        market = st.selectbox("Nearest market (where you operate)", sorted(markets["market"]))
        crops = st.multiselect("Crops you handle (leave empty for all)", cfg.TARGET_CROPS)
        c1, c2 = st.columns(2)
        capacity = c1.number_input("Capacity (kg: storage space, truck load, or most you buy)",
                                   min_value=50, value=5_000, step=50)
        price = c2.number_input("Your price (storage: UGX/kg/month; drying: UGX/kg; transport: "
                                "UGX per trip; buyer: UGX/kg you pay)", min_value=1, value=20)
        description = st.text_area("Short description", max_chars=300)
        documents = st.text_area("Documents for verification (for example trading licence number, "
                                 "UCDA/MAAIF registration, warehouse certificate)", max_chars=300)
        ok = st.form_submit_button("Submit for verification", type="primary")
    if ok:
        try:
            store.register_provider(name=name, contact=contact, kind=kind, market=market,
                                    crops=crops, capacity_kg=capacity, price=price,
                                    description=description, documents=documents)
            st.session_state.flash = ("Registration submitted. An administrator will verify it "
                                      "before farmers can see your service.")
            st.rerun()
        except ServiceError as e:
            _show_error(e)


def _request_actions(store: ServiceStore, b: pd.Series, provider_id: int) -> None:
    bid = int(b["id"])
    if b["status"] == "requested":
        with st.form(f"quote_{bid}"):
            amount = st.number_input("Your quotation (UGX, total)", min_value=0, value=0, step=1000)
            note = st.text_input("Note to the farmer (optional)")
            c1, c2 = st.columns(2)
            send = c1.form_submit_button("Send quotation", type="primary")
            decline = c2.form_submit_button("Decline request")
        if send:
            try:
                store.quote(bid, provider_id, amount, note)
                st.rerun()
            except ServiceError as e:
                _show_error(e)
        if decline:
            store.decline_request(bid, provider_id, note)
            st.rerun()
    elif b["status"] == "accepted":
        st.markdown(f"Farmer contact: **{b['farmer_contact']}**")
        with st.form(f"recv_{bid}"):
            cond = st.text_input("Condition of the produce on arrival (for example dry, 10 bags, "
                                 "no pests)")
            if st.form_submit_button("Record produce received", type="primary"):
                try:
                    store.mark_received(bid, provider_id, cond)
                    st.rerun()
                except ServiceError as e:
                    _show_error(e)
    elif b["status"] == "in_progress":
        st.markdown(f"Farmer contact: **{b['farmer_contact']}**")
        with st.form(f"upd_{bid}", clear_on_submit=True):
            note = st.text_input("Status update for the farmer")
            cond = st.text_input("Condition now (optional)")
            c1, c2 = st.columns(2)
            post = c1.form_submit_button("Post update")
            done = c2.form_submit_button("Mark completed")
        if post:
            try:
                store.post_update(bid, provider_id, note, cond)
                st.rerun()
            except ServiceError as e:
                _show_error(e)
        if done:
            store.complete(bid, provider_id, note)
            st.rerun()


def provider_portal(markets: pd.DataFrame) -> None:
    store = get_store()
    store.expire_overdue()
    contact = st.session_state.phone or st.session_state.email
    st.header("Service provider portal")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    mine = store.providers_for_contact(contact)
    if mine.empty:
        st.markdown("Register your service. An administrator checks every registration before "
                    "farmers can see it.")
        _registration_form(store, contact, markets, "register_first")
        return

    for _, p in mine.iterrows():
        pid = int(p["id"])
        avg, n = store.provider_rating(pid)
        status_text = {"pending": "Waiting for verification", "verified": "Verified",
                       "rejected": "Not approved"}[p["status"]]
        st.subheader(f"{p['name']} · {_kind_label(p['kind'])}")
        st.caption(f"{p['market']} · {_price_text(p['kind'], p['price'])} · capacity "
                   f"{p['capacity_kg']:,.0f} kg · {status_text} · {_stars(avg, n)}")
        if p["status"] == "rejected" and p["admin_note"]:
            st.warning(f"Administrator's note: {p['admin_note']}")
        if p["status"] != "verified":
            continue

        bookings = store.provider_bookings(pid)
        open_ = bookings[bookings["status"].isin(["requested", "accepted", "in_progress"])]
        if open_.empty:
            st.caption("No open requests.")

        # Shared transport groups first
        grouped = set()
        if p["kind"] == "transport":
            for g in store.shared_groups(pid):
                if len(g) < 2:
                    continue
                grouped.update(g)
                rows = bookings[bookings["id"].isin(g)]
                total = rows["quantity_kg"].sum()
                with st.container(border=True):
                    r0 = rows.iloc[0]
                    st.markdown(f"**Shared trip request** · {r0['origin_market']} → "
                                f"{r0['destination_market']} · {len(g)} farmers · {total:,.0f} kg · "
                                f"dates {rows['start_date'].min()} to {rows['start_date'].max()}")
                    with st.form(f"group_{'_'.join(map(str, g))}"):
                        price = st.number_input("Price per trip (UGX)", min_value=1,
                                                value=int(p["price"]), step=1000)
                        trips, shares = ServiceStore.split_trip_cost(price, p["capacity_kg"],
                                                                     rows["quantity_kg"].tolist())
                        st.caption(f"{trips} trip(s). Each farmer pays a share by kg: "
                                   + ", ".join(f"{s:,.0f} UGX" for s in shares))
                        if st.form_submit_button("Quote this shared trip", type="primary"):
                            try:
                                store.quote_group(g, pid, price)
                                st.rerun()
                            except ServiceError as e:
                                _show_error(e)

        for _, b in open_.iterrows():
            if int(b["id"]) in grouped:
                continue
            with st.container(border=True):
                route = (f"{b['origin_market']} → {b['destination_market']}"
                         if b["destination_market"] else b["origin_market"])
                st.markdown(f"**Request {b['id']}** · {b['quantity_kg']:,.0f} kg {b['crop']} · "
                            f"{route} · from {b['start_date']}"
                            + (f" to {b['end_date']}" if b["end_date"] else "")
                            + f" · {STATUS_LABELS[b['status']]}")
                if b["status"] == "requested":
                    due = pd.Timestamp(b["requested_at"]) + pd.Timedelta(hours=cfg.QUOTE_DEADLINE_HOURS)
                    st.caption(f"Reply by {due:%d %b %Y %H:%M} UTC or the request expires.")
                _request_actions(store, b, pid)

        done = bookings[~bookings["status"].isin(["requested", "accepted", "in_progress"])]
        if not done.empty:
            with st.expander(f"Past requests ({len(done)})"):
                st.dataframe(pd.DataFrame({
                    "Request": done["id"], "Crop": done["crop"], "kg": done["quantity_kg"],
                    "Status": done["status"].map(STATUS_LABELS), "Quote (UGX)": done["quote_ugx"],
                    "Rating": done["rating_stars"],
                }), hide_index=True, width="stretch")

    with st.expander("Register another service"):
        _registration_form(store, contact, markets, "register_more")


# =============================================================================
# Administrator
# =============================================================================
def admin_page() -> None:
    store = get_store()
    store.expire_overdue()
    st.header("Administrator")
    pending = store.providers(status="pending")
    st.subheader(f"Registrations waiting for verification ({len(pending)})")
    if pending.empty:
        st.caption("Nothing to verify.")
    for _, p in pending.iterrows():
        pid = int(p["id"])
        with st.container(border=True):
            st.markdown(f"**{p['name']}** · {_kind_label(p['kind'])} · {p['market']} · "
                        f"{_price_text(p['kind'], p['price'])} · capacity {p['capacity_kg']:,.0f} kg")
            st.caption(f"Contact: {p['contact']} · crops: {p['crops'] or 'all'} · "
                       f"registered {pd.Timestamp(p['created_at']):%d %b %Y}")
            st.markdown(f"Documents: {p['documents'] or '(none given)'}  \n"
                        f"Description: {p['description'] or '(none)'}")
            with st.form(f"verify_{pid}"):
                note = st.text_input("Note (shown to the provider if not approved)")
                c1, c2 = st.columns(2)
                ok = c1.form_submit_button("Verify", type="primary")
                no = c2.form_submit_button("Do not approve")
            if ok:
                store.set_provider_status(pid, "verified", note)
                st.rerun()
            if no:
                store.set_provider_status(pid, "rejected", note)
                st.rerun()

    st.subheader("All providers")
    allp = store.providers()
    if allp.empty:
        st.caption("No providers registered yet.")
    else:
        ratings = [store.provider_rating(int(i)) for i in allp["id"]]
        st.dataframe(pd.DataFrame({
            "Name": allp["name"], "Service": allp["kind"].map(_kind_label), "Market": allp["market"],
            "Status": allp["status"], "Price": allp["price"], "Capacity (kg)": allp["capacity_kg"],
            "Rating": [r[0] for r in ratings], "Ratings": [r[1] for r in ratings],
        }), hide_index=True, width="stretch")
        if st.checkbox("Show a verified provider's details to suspend it"):
            ver = allp[allp["status"] == "verified"]
            if not ver.empty:
                pid = st.selectbox("Provider", ver["id"].tolist(),
                                   format_func=lambda i: ver.set_index("id").at[i, "name"])
                if st.button("Suspend (mark not approved)"):
                    store.set_provider_status(int(pid), "rejected", "Suspended by administrator")
                    st.rerun()

    with st.expander("Bookings overview"):
        with store._conn() as c:
            counts = pd.read_sql_query("SELECT status, COUNT(*) AS bookings FROM bookings "
                                       "GROUP BY status", c)
        if counts.empty:
            st.caption("No bookings yet.")
        else:
            counts["status"] = counts["status"].map(STATUS_LABELS)
            st.dataframe(counts, hide_index=True)
