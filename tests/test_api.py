"""
Tests for the web API (api/). Uses a temporary services database.

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402
from api import settings  # noqa: E402
from src.services import ServiceStore  # noqa: E402

FARMER = "0772 123 456"
FARMER_NORM = "+256772123456"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-pass")
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AGRIPRICE_LLM_PROVIDER"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(settings, "SECRETS_FILE", tmp_path / "missing.toml")
    settings.file_settings.cache_clear()
    settings.app_secret.cache_clear()
    monkeypatch.setattr(main, "_store", ServiceStore(tmp_path / "api.db"))
    main._explanations.clear()
    yield TestClient(main.app)
    settings.file_settings.cache_clear()
    settings.app_secret.cache_clear()


def sign_in(client, role="farmer", contact=FARMER, **kw):
    r = client.post("/api/auth/signin", json={"role": role, "contact": contact, "consent": True, **kw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def provider(store: ServiceStore, kind="transport", market="Mbarara", price=350_000,
             capacity=5_000, contact="truck@example.com", verify=True):
    pid = store.register_provider(name=f"{market} {kind}", contact=contact, kind=kind, market=market,
                                  crops=[], capacity_kg=capacity, price=price)
    if verify:
        store.set_provider_status(pid, "verified")
    return pid


ADVICE = {"crop": "Millet", "market": "Mbarara", "quantity_kg": 500}


# ------------------------------------------------------------------ sign-in
def test_sign_in_rules(client):
    ok = client.post("/api/auth/signin", json={"role": "farmer", "contact": FARMER, "consent": True,
                                                "name": "Hillary"})
    assert ok.status_code == 200 and ok.json()["user"]["contact"] == FARMER_NORM
    assert client.post("/api/auth/signin", json={"role": "farmer", "contact": "12345",
                                                  "consent": True}).status_code == 400
    assert client.post("/api/auth/signin", json={"role": "farmer", "contact": FARMER,
                                                  "consent": False}).status_code == 400
    assert client.post("/api/auth/signin", json={"role": "admin", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/signin", json={"role": "admin", "password": "admin-pass"}).status_code == 200


def test_tokens_and_roles(client):
    h = sign_in(client)
    assert client.get("/api/auth/me", headers=h).json()["role"] == "farmer"
    tampered = {"Authorization": h["Authorization"][:-2] + "xx"}
    assert client.get("/api/auth/me", headers=tampered).status_code == 401
    assert client.post("/api/advice", json=ADVICE).status_code == 401
    ph = sign_in(client, "provider", "truck@example.com")
    assert client.post("/api/advice", json=ADVICE, headers=ph).status_code == 403
    assert client.get("/api/admin/overview", headers=h).status_code == 403


def test_broken_settings_file_is_reported(client, tmp_path, monkeypatch):
    bad = tmp_path / "secrets.toml"
    bad.write_text('Administrator sign-in\nADMIN_PASSWORD = "x"\n')
    monkeypatch.setattr(settings, "SECRETS_FILE", bad)
    settings.file_settings.cache_clear()
    r = client.post("/api/auth/signin", json={"role": "admin", "password": "x"})
    assert r.status_code == 503 and "could not be read" in r.json()["detail"]


# ------------------------------------------------------------------ data and advice
def test_meta_markets_and_location(client):
    m = client.get("/api/meta").json()
    assert "Maize" in m["crops"] and m["max_horizon"] == 6
    mk = client.get("/api/markets", params={"crop": "Millet"}).json()
    assert any(x["market"] == "Mbarara" for x in mk)
    near = client.post("/api/location/resolve", json={"crop": "Millet", "lat": -0.61, "lon": 30.66}).json()
    assert near["market"] == "Mbarara" and near["source"] == "location"
    far = client.post("/api/location/resolve", json={"crop": "Millet", "lat": 55.75, "lon": 37.62}).json()
    assert far["market"] is None and "km" in far["note"]


def test_advice_json(client):
    h = sign_in(client)
    r = client.post("/api/advice", json=ADVICE, headers=h)
    assert r.status_code == 200
    a = r.json()
    json.dumps(a, allow_nan=False)                      # strict JSON, no NaN
    assert a["advice"]["recommendation"] == "Sell now"
    assert len(a["forecast"]["points"]) == 6 and len(a["advice"]["options"]) == 7
    assert a["comparison"]["best_alternative"]["market"] == "Fort Portal"
    assert a["sell_market"] == "Fort Portal" and "Mbarara" in a["explanation_template"]
    bad = client.post("/api/advice", json={**ADVICE, "market": "Nowhere"}, headers=h)
    assert bad.status_code == 400


def test_explain_without_model_uses_template(client):
    r = client.post("/api/assistant/explain", json=ADVICE, headers=sign_in(client)).json()
    assert r["source"] == "template" and not r["ok"] and "Mbarara" in r["text"]
    assert client.post("/api/assistant/ask", json={"advice": ADVICE, "question": "Why?"},
                       headers=sign_in(client)).status_code == 503


# ------------------------------------------------------------------ services
def test_booking_lifecycle_through_the_api(client):
    s = main.store()
    pid = provider(s)
    fh = sign_in(client)
    ph = sign_in(client, "provider", "truck@example.com")
    other = sign_in(client, "provider", "other@example.com")

    opts = client.post("/api/services/options", json=ADVICE, headers=fh).json()
    transport = [g for g in opts["groups"] if g["kind"] == "transport"][0]
    assert transport["providers"][0]["id"] == pid

    start = (date.today() + timedelta(days=3)).isoformat()
    bid = client.post("/api/bookings", headers=fh, json={
        "provider_id": pid, "crop": "Millet", "quantity_kg": 500, "start_date": start,
        "origin_market": "Mbarara", "destination_market": "Fort Portal"}).json()["id"]

    mine = client.get("/api/bookings", headers=fh).json()[0]
    assert mine["status"] == "requested" and mine["provider_contact"] is None   # hidden for now

    req = client.get("/api/provider/me", headers=ph).json()["services"][0]["bookings"][0]
    assert req["farmer_contact"] is None and req["reply_by"]
    assert client.post(f"/api/provider/bookings/{bid}/quote", json={"amount_ugx": 300_000},
                       headers=other).status_code == 403                       # not their booking
    assert client.post(f"/api/provider/bookings/{bid}/quote", json={"amount_ugx": 300_000},
                       headers=ph).status_code == 200
    assert client.post(f"/api/bookings/{bid}/accept", headers=fh).status_code == 200

    mine = client.get("/api/bookings", headers=fh).json()[0]
    assert mine["provider_contact"] == "truck@example.com" and mine["map_link"].startswith("https://")
    for path, body in [("receive", {"condition": "Dry, 10 bags"}), ("update", {"note": "On the way"}),
                       ("complete", {})]:
        assert client.post(f"/api/provider/bookings/{bid}/{path}", json=body, headers=ph).status_code == 200
    assert client.post(f"/api/bookings/{bid}/rate", json={"stars": 5, "price_received_ugx_per_kg": 5400},
                       headers=fh).status_code == 200
    done = client.get("/api/bookings", headers=fh).json()[0]
    assert done["status"] == "completed" and done["rating_stars"] == 5
    assert [e["label"] for e in done["events"]][-1] == "Completed"


def test_shared_trip_and_registration_and_admin(client):
    s = main.store()
    pid = provider(s)
    start = date.today() + timedelta(days=5)
    for contact, kg in [("0701 111 111", 1000), ("0702 222 222", 3000)]:
        h = sign_in(client, contact=contact)
        client.post("/api/bookings", headers=h, json={
            "provider_id": pid, "crop": "Maize", "quantity_kg": kg, "start_date": start.isoformat(),
            "origin_market": "Mbarara", "destination_market": "Owino", "share_transport": True})
    ph = sign_in(client, "provider", "truck@example.com")
    g = client.get("/api/provider/me", headers=ph).json()["services"][0]["shared_groups"][0]
    assert g["total_kg"] == 4000
    shares = client.post("/api/provider/groups/quote", headers=ph,
                         json={"booking_ids": g["booking_ids"], "price_per_trip": 350_000}).json()["shares"]
    assert sorted(shares) == [87_500, 262_500]

    nh = sign_in(client, "provider", "new@example.com")
    assert client.post("/api/provider/register", headers=nh, json={
        "name": "New Store", "kind": "storage", "market": "Gulu", "capacity_kg": 5000,
        "price": 20}).status_code == 200
    ah = {"Authorization": "Bearer " + client.post(
        "/api/auth/signin", json={"role": "admin", "password": "admin-pass"}).json()["token"]}
    pending = client.get("/api/admin/providers", params={"status": "pending"}, headers=ah).json()
    assert [p["name"] for p in pending] == ["New Store"]
    client.post(f"/api/admin/providers/{pending[0]['id']}/status", json={"status": "verified"}, headers=ah)
    assert client.get("/api/admin/providers", params={"status": "pending"}, headers=ah).json() == []


def test_expired_request_resend(client):
    s = main.store()
    a = provider(s, kind="storage", market="Gulu", price=20, capacity=10_000, contact="a@example.com")
    b = provider(s, kind="storage", market="Gulu", price=22, capacity=10_000, contact="b@example.com")
    from datetime import datetime, timezone
    old = datetime.now(timezone.utc) - timedelta(hours=50)
    bid = s.request_booking(farmer_contact=FARMER_NORM, provider_id=a, crop="Beans", quantity_kg=800,
                            start_date=date.today(), end_date=date.today() + timedelta(days=90),
                            origin_market="Gulu", now=old)
    fh = sign_in(client)
    item = client.get("/api/bookings", headers=fh).json()[0]
    assert item["status"] == "expired" and item["next_option"]["id"] == b
    assert client.post(f"/api/bookings/{bid}/resend", headers=fh).json()["provider_name"]


def test_api_paths_never_fall_through_to_the_front_end(client):
    assert client.get("/api/nothing-here").status_code == 404
