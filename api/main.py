"""
AgriPrice Advisor web API.

Wraps the existing back end (src/) for the new web front end (frontend/), and
serves the built front end from frontend/dist at "/". Run from the project root:

    python -m uvicorn api.main:app --port 8000          # then open http://localhost:8000

Interactive API documentation is at /api/docs.

No forecasting, advice or service rule lives here: every endpoint calls a
function in src/, so the Streamlit app, the scripts and this API agree.
"""

from __future__ import annotations

import hmac
import os
from datetime import date

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import assistant, flow
from src import config as cfg
from src.advice_service import get_advice, load_context, reload_context
from src.services import STATUS_LABELS, ServiceError, ServiceStore, maps_link

from . import serialize, settings
from .auth import User, current_user, make_token, require

app = FastAPI(title="AgriPrice Advisor API", version="1.0",
              docs_url="/api/docs", openapi_url="/api/openapi.json")
FRONTEND_DIST = cfg.ROOT / "frontend" / "dist"
_store: ServiceStore | None = None


def store() -> ServiceStore:
    global _store
    if _store is None:
        # AGRIPRICE_DB lets tests and demos point at a separate database file.
        _store = ServiceStore(os.environ.get("AGRIPRICE_DB") or cfg.SERVICES_DB)
    return _store


@app.exception_handler(ServiceError)
def _service_error(_: Request, e: ServiceError):
    return JSONResponse(status_code=400, content={"detail": str(e)})


@app.exception_handler(ValueError)
def _value_error(_: Request, e: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(e)})


# =============================================================================
# Sign-in
# =============================================================================
class SignIn(BaseModel):
    role: str = Field(pattern="^(farmer|provider|admin)$")
    contact: str | None = None
    name: str = ""
    consent: bool = False
    password: str | None = None


@app.post("/api/auth/signin")
def sign_in(body: SignIn):
    if body.role == "admin":
        problem = settings.settings_problem()
        if problem:
            raise HTTPException(503, problem)
        expected = settings.get("ADMIN_PASSWORD")
        if not expected:
            raise HTTPException(503, "Administrator sign-in is not set up. Add ADMIN_PASSWORD to "
                                     ".streamlit/secrets.toml and restart.")
        if not body.password or not hmac.compare_digest(body.password, expected):
            raise HTTPException(401, "Wrong password.")
        user = User("admin", "admin", "Administrator")
    else:
        parsed = flow.parse_contact(body.contact or "")
        if parsed is None:
            if "@" in (body.contact or ""):
                raise HTTPException(400, "That email address doesn't look right. Check it, for "
                                         "example name@example.com.")
            raise HTTPException(400, "Enter a Ugandan mobile number (for example 0772 123 456) "
                                     "or an email address.")
        if not body.consent:
            raise HTTPException(400, "Please tick the consent box to continue.")
        user = User(body.role, parsed[1], body.name.strip()[:60])
    return {"token": make_token(user), "user": user.__dict__}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return user.__dict__


# =============================================================================
# Reference data
# =============================================================================
@app.get("/api/meta")
def meta():
    ctx = load_context()
    return {
        "crops": flow.available_crops(ctx.summary),
        "reference_month": ctx.summary.attrs.get("reference_month") or str(
            ctx.summary["last_observed_month"].max())[:7],
        "max_horizon": cfg.MAX_FORECAST_HORIZON,
        "service_kinds": cfg.SERVICE_KINDS,
        "quote_deadline_hours": cfg.QUOTE_DEADLINE_HOURS,
        "assistant": assistant_status(),
        "all_markets": sorted(ctx.markets["market"]),
    }


@app.get("/api/markets")
def markets(crop: str):
    ctx = load_context()
    names = flow.available_markets(ctx.summary, crop)
    m = ctx.markets[ctx.markets["market"].isin(names)]
    return serialize.records(m[["market", "region", "latitude", "longitude"]])


@app.get("/api/costs/{crop}")
def default_costs(crop: str):
    from src.decision import Costs
    c = Costs.default_for(crop)
    return {"storage_ugx_per_kg_month": c.storage_ugx_per_kg_month,
            "loss_pct_per_month": c.loss_rate_per_month * 100,
            "transport_ugx_per_kg_km": cfg.TRANSPORT_UGX_PER_KG_KM}


class Locate(BaseModel):
    crop: str
    lat: float | None = None
    lon: float | None = None
    manual: str | None = None


@app.post("/api/location/resolve")
def resolve_location(body: Locate):
    ctx = load_context()
    coords = (body.lat, body.lon) if body.lat is not None and body.lon is not None else None
    c = flow.resolve_market(body.crop, ctx.summary, ctx.markets, coords, body.manual)
    return serialize.clean(c.__dict__)


# =============================================================================
# Advice
# =============================================================================
class AdviceRequest(BaseModel):
    crop: str
    market: str
    quantity_kg: float = Field(gt=0, le=1_000_000)
    horizon: int = Field(default=cfg.MAX_FORECAST_HORIZON, ge=1, le=cfg.MAX_FORECAST_HORIZON)
    storage: float | None = Field(default=None, ge=0, le=10_000)
    loss_pct: float | None = Field(default=None, ge=0, le=50)
    transport_rate: float | None = Field(default=None, ge=0, le=1_000)

    def run(self):
        return get_advice(**self.model_dump())


@app.post("/api/advice")
def advice(body: AdviceRequest, user: User = Depends(require("farmer"))):
    return serialize.advice(body.run())


# =============================================================================
# AI assistant
# =============================================================================
def _llm():
    return assistant.get_llm(settings.llm_settings())


def assistant_status() -> dict:
    llm = _llm()
    return {"enabled": llm is not None, "model": getattr(llm, "model", None)}


_explanations: dict = {}


@app.post("/api/assistant/explain")
def explain(body: AdviceRequest, user: User = Depends(require("farmer"))):
    r = body.run()
    llm = _llm()
    if llm is None:
        return {"ok": False, "text": r.template_explanation(), "source": "template",
                "reason": "no language model configured"}
    key = (body.model_dump_json(), llm.name, llm.model)
    if key not in _explanations:
        _explanations[key] = assistant.explain(llm, r.payload(),
                                               known_markets=load_context().markets["market"].tolist())
    reply = _explanations[key]
    if reply.ok:
        return {"ok": True, "text": reply.text, "source": "ai", "model": llm.model}
    return {"ok": False, "text": r.template_explanation(), "source": "template",
            "reason": reply.reason}


class Ask(BaseModel):
    advice: AdviceRequest
    question: str = Field(min_length=1, max_length=500)
    history: list[dict] = []


@app.post("/api/assistant/ask")
def ask(body: Ask, user: User = Depends(require("farmer"))):
    llm = _llm()
    if llm is None:
        raise HTTPException(503, "The question assistant needs a language model. Add an API key "
                                 "in .streamlit/secrets.toml.")
    history = [h for h in body.history if h.get("role") in ("user", "assistant")][-6:]
    reply = assistant.answer(llm, body.advice.run().payload(), body.question, history,
                             known_markets=load_context().markets["market"].tolist())
    text = reply.text or "Sorry, I could not answer that just now. Please try again in a moment."
    return {"ok": reply.ok, "text": text}


# =============================================================================
# Services: options for a farmer
# =============================================================================
def _option_groups(r) -> list[tuple[str, str]]:
    if r.adv.months > 0:
        return [("storage", f"Storage near {r.market} for {r.adv.months} month"
                            f"{'s' if r.adv.months > 1 else ''}"),
                ("preservation", "Drying and treatment before storage")]
    groups = []
    if r.sell_market != r.market:
        groups.append(("transport", f"Transport from {r.market} to {r.sell_market}"))
    groups.append(("buyer", f"Buyers for your {r.crop.lower()}"))
    return groups


def _ranked(kind: str, crop: str, qty: float, market: str, months: int, rate: float) -> list[dict]:
    df = store().rank_options(kind=kind, crop=crop, quantity_kg=qty, home_market=market,
                              markets=load_context().markets, months=months,
                              transport_ugx_per_kg_km=rate)
    if df.empty:
        return []
    keep = ["id", "name", "kind", "market", "crops", "capacity_kg", "price", "description",
            "distance_km", "estimate_ugx", "rating", "rating_count"] + (["trips"] if "trips" in df else [])
    return serialize.records(df[keep])


@app.post("/api/services/options")
def service_options(body: AdviceRequest, user: User = Depends(require("farmer"))):
    r = body.run()
    rate = r.costs.transport_ugx_per_kg_km
    groups = [{"kind": k, "heading": h,
               "providers": _ranked(k, r.crop, r.quantity_kg, r.market, r.adv.months, rate)}
              for k, h in _option_groups(r)]
    used = {g["kind"] for g in groups}
    return {"recommendation": r.adv.recommendation, "months": r.adv.months,
            "market": r.market, "sell_market": r.sell_market, "groups": groups,
            "other_kinds": [k for k in cfg.SERVICE_KINDS if k not in used]}


@app.get("/api/services/{kind}")
def services_of_kind(kind: str, crop: str, quantity_kg: float, market: str, months: int = 1,
                     user: User = Depends(require("farmer"))):
    if kind not in cfg.SERVICE_KINDS:
        raise HTTPException(404, "Unknown service type.")
    return _ranked(kind, crop, quantity_kg, market, max(months, 1), cfg.TRANSPORT_UGX_PER_KG_KM)


@app.get("/api/transport/sharing-estimate")
def sharing_estimate(provider_id: int, origin: str, destination: str, start_date: date,
                     quantity_kg: float, user: User = Depends(require("farmer"))):
    return serialize.clean(store().sharing_estimate(provider_id, origin, destination,
                                                    start_date, quantity_kg))


# =============================================================================
# Bookings (farmer)
# =============================================================================
class BookingRequest(BaseModel):
    provider_id: int
    crop: str
    quantity_kg: float = Field(gt=0)
    start_date: date
    end_date: date | None = None
    origin_market: str
    destination_market: str | None = None
    share_transport: bool = False
    advice_snapshot: dict = {}


@app.post("/api/bookings")
def create_booking(body: BookingRequest, user: User = Depends(require("farmer"))):
    bid = store().request_booking(farmer_contact=user.contact, **body.model_dump())
    return {"id": bid}


def _booking_json(b: pd.Series | dict, viewer: str) -> dict:
    b = dict(b)
    s = store()
    visible = s.contacts_visible(b)
    mk = load_context().markets.set_index("market")
    out = {k: b.get(k) for k in [
        "id", "kind", "crop", "quantity_kg", "start_date", "end_date", "origin_market",
        "destination_market", "share_transport", "status", "quote_ugx", "quote_note",
        "requested_at", "provider_id", "provider_name", "provider_market",
        "rating_stars", "price_received_ugx_per_kg"]}
    out["status_label"] = STATUS_LABELS.get(b["status"], b["status"])
    out["share_transport"] = bool(b.get("share_transport"))
    out["contacts_visible"] = visible
    if viewer == "farmer":
        out["provider_contact"] = b.get("provider_contact") if visible else None
        pm = b.get("provider_market")
        out["map_link"] = (maps_link(mk.at[pm, "latitude"], mk.at[pm, "longitude"])
                           if visible and pm in mk.index else None)
    else:
        out["farmer_contact"] = b.get("farmer_contact") if visible else None
        out["reply_by"] = (pd.Timestamp(b["requested_at"])
                           + pd.Timedelta(hours=cfg.QUOTE_DEADLINE_HOURS)).isoformat()
    ev = s.events(int(b["id"]))
    out["events"] = [{"at": e["at"], "actor": e["actor"],
                      "status": e["status"] if isinstance(e["status"], str) else None,
                      "label": STATUS_LABELS.get(e["status"], "Update") if isinstance(e["status"], str) else "Update",
                      "note": e["note"], "condition": e["condition"]} for _, e in ev.iterrows()]
    return serialize.clean(out)


@app.get("/api/bookings")
def my_bookings(user: User = Depends(require("farmer"))):
    s = store()
    s.expire_overdue()
    df = s.farmer_bookings(user.contact)
    out = []
    for _, b in df.iterrows():
        item = _booking_json(b, "farmer")
        if b["status"] in ("expired", "declined_by_provider"):
            nxt = s.next_ranked(int(b["id"]), load_context().markets)
            item["next_option"] = (serialize.records(nxt.head(1)[["id", "name", "market", "kind", "price"]])[0]
                                   if not nxt.empty else None)
        if (b["kind"] == "storage" and b["status"] in ("accepted", "in_progress") and b["end_date"]
                and (date.fromisoformat(b["end_date"]) - date.today()).days <= 30):
            try:
                r = get_advice(crop=b["crop"], market=b["origin_market"], quantity_kg=b["quantity_kg"])
                item["recheck"] = {"recommendation": r.adv.recommendation,
                                   "price": r.fc.origin_price, "date": r.fc.origin_date,
                                   "confidence_label": r.adv.confidence_label}
            except ValueError:
                pass
        out.append(serialize.clean(item))
    return out


class Note(BaseModel):
    note: str = ""


@app.post("/api/bookings/{bid}/accept")
def accept(bid: int, user: User = Depends(require("farmer"))):
    store().accept_quote(bid, user.contact)
    return {"ok": True}


@app.post("/api/bookings/{bid}/decline")
def decline(bid: int, body: Note = Note(), user: User = Depends(require("farmer"))):
    store().decline_quote(bid, user.contact, body.note)
    return {"ok": True}


@app.post("/api/bookings/{bid}/cancel")
def cancel(bid: int, body: Note = Note(), user: User = Depends(require("farmer"))):
    store().cancel(bid, user.contact, body.note)
    return {"ok": True}


@app.post("/api/bookings/{bid}/resend")
def resend(bid: int, user: User = Depends(require("farmer"))):
    """Send an expired or declined request to the next ranked provider."""
    s = store()
    b = s.booking(bid)
    if b["farmer_contact"] != user.contact:
        raise HTTPException(403, "This booking belongs to another farmer.")
    nxt = s.next_ranked(bid, load_context().markets)
    if nxt.empty:
        raise HTTPException(404, "No other verified provider is available right now.")
    new_id = s.request_booking(
        farmer_contact=user.contact, provider_id=int(nxt.iloc[0]["id"]), crop=b["crop"],
        quantity_kg=b["quantity_kg"], start_date=date.fromisoformat(b["start_date"]),
        end_date=date.fromisoformat(b["end_date"]) if b["end_date"] else None,
        origin_market=b["origin_market"], destination_market=b["destination_market"],
        share_transport=bool(b["share_transport"]))
    return {"id": new_id, "provider_name": nxt.iloc[0]["name"]}


class Rating(BaseModel):
    stars: int = Field(ge=1, le=5)
    comment: str = ""
    price_received_ugx_per_kg: float | None = Field(default=None, ge=0)


@app.post("/api/bookings/{bid}/rate")
def rate(bid: int, body: Rating, user: User = Depends(require("farmer"))):
    store().rate(bid, user.contact, body.stars, body.comment, body.price_received_ugx_per_kg or None)
    return {"ok": True}


# =============================================================================
# Provider portal
# =============================================================================
def _my_provider_ids(user: User) -> set[int]:
    return set(store().providers_for_contact(user.contact)["id"].astype(int))


def _own(user: User, provider_id: int) -> int:
    if provider_id not in _my_provider_ids(user):
        raise HTTPException(403, "This service belongs to another account.")
    return provider_id


@app.get("/api/provider/me")
def provider_me(user: User = Depends(require("provider"))):
    s = store()
    s.expire_overdue()
    services = []
    for _, p in s.providers_for_contact(user.contact).iterrows():
        pid = int(p["id"])
        avg, n = s.provider_rating(pid)
        bookings = s.provider_bookings(pid) if p["status"] == "verified" else pd.DataFrame()
        groups = []
        if p["kind"] == "transport" and p["status"] == "verified":
            for g in s.shared_groups(pid):
                if len(g) > 1:
                    rows = bookings[bookings["id"].isin(g)]
                    trips, shares = ServiceStore.split_trip_cost(p["price"], p["capacity_kg"],
                                                                 rows["quantity_kg"].tolist())
                    groups.append({"booking_ids": g, "total_kg": float(rows["quantity_kg"].sum()),
                                   "origin": rows.iloc[0]["origin_market"],
                                   "destination": rows.iloc[0]["destination_market"],
                                   "from": rows["start_date"].min(), "to": rows["start_date"].max(),
                                   "suggested_trips": trips, "suggested_shares": shares})
        services.append(serialize.clean({
            **{k: p[k] for k in ["id", "name", "kind", "market", "crops", "capacity_kg", "price",
                                 "description", "status", "admin_note", "created_at"]},
            "rating": avg, "rating_count": n,
            "bookings": [_booking_json(b, "provider") for _, b in bookings.iterrows()],
            "shared_groups": groups,
        }))
    return {"services": services}


class Registration(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    kind: str
    market: str
    crops: list[str] = []
    capacity_kg: float = Field(gt=0)
    price: float = Field(gt=0)
    description: str = Field(default="", max_length=300)
    documents: str = Field(default="", max_length=300)


@app.post("/api/provider/register")
def register(body: Registration, user: User = Depends(require("provider"))):
    if body.market not in set(load_context().markets["market"]):
        raise HTTPException(400, "Choose a market from the list.")
    pid = store().register_provider(contact=user.contact, **body.model_dump())
    return {"id": pid}


class Quote(BaseModel):
    amount_ugx: float = Field(gt=0)
    note: str = ""


def _booking_provider(bid: int, user: User) -> int:
    return _own(user, int(store().booking(bid)["provider_id"]))


@app.post("/api/provider/bookings/{bid}/quote")
def quote(bid: int, body: Quote, user: User = Depends(require("provider"))):
    store().quote(bid, _booking_provider(bid, user), body.amount_ugx, body.note)
    return {"ok": True}


@app.post("/api/provider/bookings/{bid}/decline")
def provider_decline(bid: int, body: Note = Note(), user: User = Depends(require("provider"))):
    store().decline_request(bid, _booking_provider(bid, user), body.note)
    return {"ok": True}


class Received(BaseModel):
    condition: str = Field(min_length=1, max_length=300)
    note: str = ""


@app.post("/api/provider/bookings/{bid}/receive")
def receive(bid: int, body: Received, user: User = Depends(require("provider"))):
    store().mark_received(bid, _booking_provider(bid, user), body.condition, body.note)
    return {"ok": True}


class Update(BaseModel):
    note: str = Field(min_length=1, max_length=300)
    condition: str = ""


@app.post("/api/provider/bookings/{bid}/update")
def update(bid: int, body: Update, user: User = Depends(require("provider"))):
    store().post_update(bid, _booking_provider(bid, user), body.note, body.condition)
    return {"ok": True}


@app.post("/api/provider/bookings/{bid}/complete")
def complete(bid: int, body: Note = Note(), user: User = Depends(require("provider"))):
    store().complete(bid, _booking_provider(bid, user), body.note)
    return {"ok": True}


class GroupQuote(BaseModel):
    booking_ids: list[int] = Field(min_length=2)
    price_per_trip: float = Field(gt=0)
    note: str = ""


@app.post("/api/provider/groups/quote")
def group_quote(body: GroupQuote, user: User = Depends(require("provider"))):
    pids = {int(store().booking(b)["provider_id"]) for b in body.booking_ids}
    if len(pids) != 1:
        raise HTTPException(400, "These requests belong to different services.")
    pid = _own(user, pids.pop())
    shares = store().quote_group(body.booking_ids, pid, body.price_per_trip, body.note)
    return {"shares": serialize.clean(shares)}


# =============================================================================
# Administrator
# =============================================================================
@app.get("/api/admin/providers")
def admin_providers(status: str | None = None, user: User = Depends(require("admin"))):
    s = store()
    df = s.providers(status=status)
    out = serialize.records(df)
    for row in out:
        row["rating"], row["rating_count"] = s.provider_rating(int(row["id"]))
    return out


class StatusChange(BaseModel):
    status: str = Field(pattern="^(verified|rejected|pending)$")
    note: str = ""


@app.post("/api/admin/providers/{pid}/status")
def admin_set_status(pid: int, body: StatusChange, user: User = Depends(require("admin"))):
    store().set_provider_status(pid, body.status, body.note)
    return {"ok": True}


@app.get("/api/admin/overview")
def admin_overview(user: User = Depends(require("admin"))):
    s = store()
    s.expire_overdue()
    with s._conn() as c:
        counts = pd.read_sql_query("SELECT status, COUNT(*) AS n FROM bookings GROUP BY status", c)
        prov = pd.read_sql_query("SELECT status, COUNT(*) AS n FROM providers GROUP BY status", c)
    return {"bookings": {STATUS_LABELS.get(r.status, r.status): int(r.n) for r in counts.itertuples()},
            "providers": {r.status: int(r.n) for r in prov.itertuples()}}


@app.post("/api/admin/reload-data")
def admin_reload(user: User = Depends(require("admin"))):
    """Pick up rebuilt data or newly selected models without restarting."""
    ctx = reload_context()
    return {"ok": True, "series": int(len(ctx.summary))}


# =============================================================================
# Front end
# =============================================================================
@app.get("/api/health")
def health():
    return {"ok": True, "frontend_built": (FRONTEND_DIST / "index.html").exists(),
            "settings_problem": settings.settings_problem()}


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    """Serve the built front end; every non-API path loads the single-page app."""
    if path.startswith("api/"):
        raise HTTPException(404, "Not found.")
    candidate = FRONTEND_DIST / path
    if path and candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        return JSONResponse(status_code=503, content={
            "detail": "The web front end has not been built. Run: cd frontend && npm install && npm run build"})
    return FileResponse(index)
