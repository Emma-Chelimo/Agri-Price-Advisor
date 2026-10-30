"""
Tests for the service platform (Phase 9). Each test uses a fresh database.

Run from the project root:   python -m pytest -q
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services import ServiceError, ServiceStore  # noqa: E402

T0 = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)
FARMER = "+256772123456"
MARKETS = pd.DataFrame({
    "market": ["Lira", "Gulu", "Owino"],
    "region": ["Northern", "Northern", "Central"],
    "latitude": [2.23, 2.78, 0.32], "longitude": [32.91, 32.30, 32.57],
})


@pytest.fixture
def store(tmp_path):
    return ServiceStore(tmp_path / "test.db")


def add(store, name, kind, market, price, capacity=10_000, crops=("Maize",), verify=True):
    pid = store.register_provider(name=name, contact=f"{name.lower()}@example.com", kind=kind,
                                  market=market, crops=list(crops), capacity_kg=capacity,
                                  price=price, documents="Trading licence", now=T0)
    if verify:
        store.set_provider_status(pid, "verified", now=T0)
    return pid


def book(store, pid, qty=500, **kw):
    args = dict(farmer_contact=FARMER, provider_id=pid, crop="Maize", quantity_kg=qty,
                start_date=date(2026, 5, 5), end_date=date(2026, 8, 5), origin_market="Lira",
                now=T0)
    args.update(kw)
    return store.request_booking(**args)


# ------------------------------------------------------------------ registry
def test_registration_and_verification(store):
    pid = add(store, "Store A", "storage", "Lira", 20, verify=False)
    assert store.provider(pid)["status"] == "pending"
    with pytest.raises(ServiceError):                    # same contact, same kind
        add(store, "Store A", "storage", "Lira", 20, verify=False)
    store.set_provider_status(pid, "verified", now=T0)
    assert store.provider(pid)["status"] == "verified"


def test_unverified_providers_are_not_ranked_or_bookable(store):
    pid = add(store, "Store A", "storage", "Lira", 20, verify=False)
    assert store.rank_options(kind="storage", crop="Maize", quantity_kg=500,
                              home_market="Lira", markets=MARKETS, months=3).empty
    with pytest.raises(ServiceError):
        book(store, pid)


def test_ranking_by_total_cost_crop_and_capacity(store):
    near = add(store, "Near", "storage", "Lira", 25)                 # 25*500*3 = 37,500
    far = add(store, "Far", "storage", "Gulu", 10)                   # 15,000 + transport
    add(store, "Small", "storage", "Lira", 5, capacity=100)          # too small
    add(store, "Beans only", "storage", "Lira", 5, crops=("Beans",))
    r = store.rank_options(kind="storage", crop="Maize", quantity_kg=500, home_market="Lira",
                           markets=MARKETS, months=3, transport_ugx_per_kg_km=3.0)
    assert r["id"].tolist() == [near, far]
    assert r.iloc[0]["estimate_ugx"] == 37_500
    buyers = [add(store, "Buyer low", "buyer", "Lira", 1000), add(store, "Buyer high", "buyer", "Lira", 1200)]
    rb = store.rank_options(kind="buyer", crop="Maize", quantity_kg=500, home_market="Lira", markets=MARKETS)
    assert rb["id"].tolist() == buyers[::-1]                          # highest earnings first


# ------------------------------------------------------------------ booking lifecycle
def test_full_lifecycle_and_contacts(store):
    pid = add(store, "Store A", "storage", "Lira", 20)
    bid = book(store, pid)
    assert not store.contacts_visible(store.booking(bid))
    store.quote(bid, pid, 30_000, "3 months", now=T0 + timedelta(hours=5))
    store.accept_quote(bid, FARMER, now=T0 + timedelta(hours=6))
    assert store.contacts_visible(store.booking(bid))
    store.mark_received(bid, pid, condition="Dry, clean, 10 bags", now=T0 + timedelta(days=4))
    store.post_update(bid, pid, "Fumigated", now=T0 + timedelta(days=30))
    store.complete(bid, pid, now=T0 + timedelta(days=95))
    store.rate(bid, FARMER, 4, "Good", price_received_ugx_per_kg=2100, now=T0 + timedelta(days=96))
    ev = store.events(bid)
    assert ev["status"].fillna("update").tolist() == [
        "requested", "quoted", "accepted", "in_progress", "update", "completed"]
    assert ev.loc[ev["status"] == "in_progress", "condition"].iloc[0] == "Dry, clean, 10 bags"
    assert store.provider_rating(pid) == (4.0, 1)
    with pytest.raises(ServiceError):
        store.rate(bid, FARMER, 5)                                    # only once


def test_wrong_person_or_wrong_state_is_refused(store):
    a = add(store, "Store A", "storage", "Lira", 20)
    b = add(store, "Store B", "storage", "Lira", 20)
    bid = book(store, a)
    with pytest.raises(ServiceError):
        store.quote(bid, b, 10_000)                                   # another provider
    with pytest.raises(ServiceError):
        store.accept_quote(bid, FARMER)                               # nothing quoted yet
    store.quote(bid, a, 10_000)
    with pytest.raises(ServiceError):
        store.accept_quote(bid, "+256700000000")                      # another farmer
    with pytest.raises(ServiceError):
        store.rate(bid, FARMER, 5)                                    # not completed
    with pytest.raises(ServiceError):
        book(store, a, qty=50_000)                                    # over capacity


def test_48_hour_rule_and_next_ranked(store):
    a = add(store, "Store A", "storage", "Lira", 20)
    b = add(store, "Store B", "storage", "Lira", 22)
    bid = book(store, a)
    assert store.expire_overdue(now=T0 + timedelta(hours=47)) == []
    assert store.expire_overdue(now=T0 + timedelta(hours=49)) == [bid]
    assert store.booking(bid)["status"] == "expired"
    nxt = store.next_ranked(bid, MARKETS)
    assert nxt["id"].tolist() == [b]                                  # the one not yet tried


# ------------------------------------------------------------------ shared transport
def test_split_trip_cost():
    trips, shares = ServiceStore.split_trip_cost(300_000, 5_000, [1_000, 3_000])
    assert trips == 1 and shares == [75_000, 225_000]
    trips, shares = ServiceStore.split_trip_cost(300_000, 5_000, [4_000, 3_000])
    assert trips == 2 and sum(shares) == 600_000


def test_shared_groups_and_group_quote(store):
    t = add(store, "Truck", "transport", "Lira", 300_000, capacity=5_000)
    common = dict(destination_market="Owino", share_transport=True, end_date=None)
    b1 = book(store, t, qty=1_000, start_date=date(2026, 5, 5), **common)
    b2 = book(store, t, qty=3_000, start_date=date(2026, 5, 7), **common)
    b3 = book(store, t, qty=500, start_date=date(2026, 5, 20), **common)      # too far apart
    b4 = book(store, t, qty=500, start_date=date(2026, 5, 6),
              destination_market="Gulu", share_transport=True, end_date=None)  # other route
    groups = store.shared_groups(t)
    assert sorted(map(sorted, groups)) == sorted([[b1, b2], [b3], [b4]])
    est = store.sharing_estimate(t, "Lira", "Owino", date(2026, 5, 6), 1_000)
    assert est["others"] == 2 and est["shared_ugx"] < est["alone_ugx"]
    shares = store.quote_group([b1, b2], t, 300_000)
    assert shares == [75_000, 225_000]
    assert store.booking(b1)["quote_ugx"] == 75_000 and "Shared trip" in store.booking(b1)["quote_note"]
    with pytest.raises(ServiceError):
        store.quote_group([b3, b4], t, 300_000)                       # different routes


def test_transport_needs_destination(store):
    t = add(store, "Truck", "transport", "Lira", 300_000, capacity=5_000)
    with pytest.raises(ServiceError):
        book(store, t, end_date=None)
