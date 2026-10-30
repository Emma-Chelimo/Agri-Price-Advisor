"""
Service platform (Phase 9, Objectives 5 and 6).

Follows Part 2 of the approved user flow:

  provider registers  ->  administrator verifies  ->  provider joins the registry
  farmer accepts advice  ->  ranked options from the registry
  farmer chooses an option  ->  booking request (crop, quantity, dates, location)
  provider replies within 48 hours with a quotation (or declines)
      no reply in 48 hours  ->  request expires, next ranked provider offered
  farmer accepts the quotation  ->  booking confirmed, contacts and map shared
  provider receives the produce, records its condition, posts status updates
  the app re-checks the advice before the planned sale date (in the UI)
  after the sale the farmer rates the provider and records the price received

Shared transport (Objective 6): transport requests that opt in to sharing, on
the same route with the same provider within SHARED_TRIP_WINDOW_DAYS, form a
group. The provider quotes one price per trip for the group and each farmer
pays a share in proportion to their kilograms.

Everything is stored in one SQLite file (config.SERVICES_DB). Every function
takes an optional `now` so the 48-hour rule can be tested.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .decision import haversine_km

KINDS = tuple(cfg.SERVICE_KINDS)
PROVIDER_STATUSES = ("pending", "verified", "rejected")

# Booking statuses and who may move a booking from one to the next.
TRANSITIONS = {
    "requested": {"quoted": "provider", "declined_by_provider": "provider",
                  "expired": "system", "cancelled": "farmer"},
    "quoted": {"accepted": "farmer", "quote_declined": "farmer", "cancelled": "farmer"},
    "accepted": {"in_progress": "provider", "cancelled": "farmer"},
    "in_progress": {"completed": "provider"},
}
ACTIVE = ("requested", "quoted", "accepted", "in_progress")
STATUS_LABELS = {
    "requested": "Waiting for quotation",
    "quoted": "Quotation received",
    "accepted": "Confirmed",
    "in_progress": "In progress",
    "completed": "Completed",
    "declined_by_provider": "Declined by provider",
    "quote_declined": "You declined the quotation",
    "expired": "No reply within 48 hours",
    "cancelled": "Cancelled",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    contact       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    market        TEXT NOT NULL,
    crops         TEXT NOT NULL DEFAULT '',     -- comma list; empty = all crops
    capacity_kg   REAL NOT NULL,                -- storage space, truck load, or most a buyer takes
    price         REAL NOT NULL,                -- unit depends on kind (config.SERVICE_KINDS)
    description   TEXT NOT NULL DEFAULT '',
    documents     TEXT NOT NULL DEFAULT '',     -- what was submitted for verification
    status        TEXT NOT NULL DEFAULT 'pending',
    admin_note    TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    verified_at   TEXT,
    UNIQUE (contact, kind)
);
CREATE TABLE IF NOT EXISTS bookings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_contact     TEXT NOT NULL,
    provider_id        INTEGER NOT NULL REFERENCES providers(id),
    kind               TEXT NOT NULL,
    crop               TEXT NOT NULL,
    quantity_kg        REAL NOT NULL,
    start_date         TEXT NOT NULL,
    end_date           TEXT,
    origin_market      TEXT NOT NULL,
    destination_market TEXT,
    share_transport    INTEGER NOT NULL DEFAULT 0,
    group_key          TEXT,
    status             TEXT NOT NULL,
    quote_ugx          REAL,
    quote_note         TEXT NOT NULL DEFAULT '',
    requested_at       TEXT NOT NULL,
    quoted_at          TEXT,
    decided_at         TEXT,
    advice_snapshot    TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS booking_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id  INTEGER NOT NULL REFERENCES bookings(id),
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    status      TEXT,
    note        TEXT NOT NULL DEFAULT '',
    condition   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS ratings (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id              INTEGER NOT NULL UNIQUE REFERENCES bookings(id),
    provider_id             INTEGER NOT NULL REFERENCES providers(id),
    stars                   INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
    comment                 TEXT NOT NULL DEFAULT '',
    price_received_ugx_per_kg REAL,
    at                      TEXT NOT NULL
);
"""


class ServiceError(ValueError):
    """An action that is not allowed (wrong person, wrong state, bad input)."""


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _iso(d: datetime | date | None) -> str | None:
    return None if d is None else d.isoformat()


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.5f},{lon:.5f}"


class ServiceStore:
    def __init__(self, path: Path | str = cfg.SERVICES_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _df(self, sql: str, params=()) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=params)

    def _one(self, sql: str, params=()) -> dict | None:
        with self._conn() as c:
            r = c.execute(sql, params).fetchone()
            return dict(r) if r else None

    # ------------------------------------------------------------------ providers
    def register_provider(self, *, name: str, contact: str, kind: str, market: str,
                          crops: list[str] | None, capacity_kg: float, price: float,
                          description: str = "", documents: str = "",
                          now: datetime | None = None) -> int:
        if kind not in KINDS:
            raise ServiceError(f"Unknown service type '{kind}'.")
        if not name.strip():
            raise ServiceError("Please enter the business name.")
        if capacity_kg <= 0 or price <= 0:
            raise ServiceError("Capacity and price must be greater than zero.")
        try:
            with self._conn() as c:
                cur = c.execute(
                    "INSERT INTO providers (name, contact, kind, market, crops, capacity_kg, price, "
                    "description, documents, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name.strip(), contact, kind, market, ",".join(crops or []), float(capacity_kg),
                     float(price), description.strip(), documents.strip(), _iso(_now(now))))
                return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            raise ServiceError(f"This contact already has a registered {kind} service.") from None

    def provider(self, provider_id: int) -> dict:
        p = self._one("SELECT * FROM providers WHERE id = ?", (provider_id,))
        if not p:
            raise ServiceError("Provider not found.")
        return p

    def providers(self, status: str | None = None, kind: str | None = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM providers WHERE 1=1", []
        if status:
            sql += " AND status = ?"; params.append(status)
        if kind:
            sql += " AND kind = ?"; params.append(kind)
        return self._df(sql + " ORDER BY created_at", params)

    def providers_for_contact(self, contact: str) -> pd.DataFrame:
        return self._df("SELECT * FROM providers WHERE contact = ? ORDER BY kind", (contact,))

    def set_provider_status(self, provider_id: int, status: str, note: str = "",
                            now: datetime | None = None) -> None:
        if status not in PROVIDER_STATUSES:
            raise ServiceError(f"Unknown status '{status}'.")
        self.provider(provider_id)
        with self._conn() as c:
            c.execute("UPDATE providers SET status = ?, admin_note = ?, verified_at = ? WHERE id = ?",
                      (status, note, _iso(_now(now)) if status == "verified" else None, provider_id))

    def update_provider(self, provider_id: int, **fields) -> None:
        allowed = {"name", "market", "crops", "capacity_kg", "price", "description"}
        bad = set(fields) - allowed
        if bad:
            raise ServiceError(f"Cannot change {', '.join(sorted(bad))}.")
        if "crops" in fields and isinstance(fields["crops"], list):
            fields["crops"] = ",".join(fields["crops"])
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE providers SET {sets} WHERE id = ?", (*fields.values(), provider_id))

    def provider_rating(self, provider_id: int) -> tuple[float | None, int]:
        r = self._one("SELECT AVG(stars) a, COUNT(*) n FROM ratings WHERE provider_id = ?",
                      (provider_id,))
        return (round(r["a"], 1) if r["a"] is not None else None, int(r["n"]))

    # ------------------------------------------------------------------ ranking
    def rank_options(self, *, kind: str, crop: str, quantity_kg: float, home_market: str,
                     markets: pd.DataFrame, months: int = 0,
                     transport_ugx_per_kg_km: float = cfg.TRANSPORT_UGX_PER_KG_KM,
                     exclude: list[int] | None = None) -> pd.DataFrame:
        """
        Verified providers that can take this crop and quantity, ranked for the farmer.

        storage       cost = price x kg x months + transport to the store
        preservation  cost = price x kg + transport to the provider
        transport     cost = price per trip x trips needed
        buyer         earnings = price x kg - transport to the buyer (higher is better)

        Ties are broken by average rating, then distance.
        """
        p = self.providers(status="verified", kind=kind)
        if exclude:
            p = p[~p["id"].isin(exclude)]
        if p.empty:
            return p
        p = p[(p["crops"] == "") | p["crops"].apply(lambda s: crop in s.split(","))]
        if kind != "transport":
            p = p[p["capacity_kg"] >= quantity_kg]
        if p.empty:
            return p
        m = markets.set_index("market")
        home = m.loc[home_market]
        coords = m.reindex(p["market"])
        p = p.copy()
        p["distance_km"] = np.round(haversine_km(home["latitude"], home["longitude"],
                                                 coords["latitude"].to_numpy(),
                                                 coords["longitude"].to_numpy()) * cfg.ROAD_FACTOR)
        p["latitude"], p["longitude"] = coords["latitude"].to_numpy(), coords["longitude"].to_numpy()
        move = transport_ugx_per_kg_km * p["distance_km"] * quantity_kg
        if kind == "storage":
            p["estimate_ugx"] = p["price"] * quantity_kg * max(months, 1) + move
        elif kind == "preservation":
            p["estimate_ugx"] = p["price"] * quantity_kg + move
        elif kind == "transport":
            p["trips"] = np.ceil(quantity_kg / p["capacity_kg"]).astype(int)
            p["estimate_ugx"] = p["price"] * p["trips"]
        else:  # buyer
            p["estimate_ugx"] = p["price"] * quantity_kg - move
        ratings = [self.provider_rating(int(i)) for i in p["id"]]
        p["rating"] = [r[0] for r in ratings]
        p["rating_count"] = [r[1] for r in ratings]
        p["_r"] = p["rating"].fillna(0)
        better_high = kind == "buyer"
        p = p.sort_values(["estimate_ugx", "_r", "distance_km"],
                          ascending=[not better_high, False, True]).drop(columns="_r")
        return p.reset_index(drop=True)

    # ------------------------------------------------------------------ bookings
    def _log(self, c, booking_id: int, actor: str, status: str | None, note: str = "",
             condition: str = "", now: datetime | None = None) -> None:
        c.execute("INSERT INTO booking_events (booking_id, at, actor, status, note, condition) "
                  "VALUES (?,?,?,?,?,?)",
                  (booking_id, _iso(_now(now)), actor, status, note, condition))

    def booking(self, booking_id: int) -> dict:
        b = self._one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
        if not b:
            raise ServiceError("Booking not found.")
        return b

    def request_booking(self, *, farmer_contact: str, provider_id: int, crop: str,
                        quantity_kg: float, start_date: date, end_date: date | None,
                        origin_market: str, destination_market: str | None = None,
                        share_transport: bool = False, advice_snapshot: dict | None = None,
                        now: datetime | None = None) -> int:
        p = self.provider(provider_id)
        if p["status"] != "verified":
            raise ServiceError("This provider is not verified yet.")
        if quantity_kg <= 0:
            raise ServiceError("Quantity must be greater than zero.")
        if p["kind"] != "transport" and quantity_kg > p["capacity_kg"]:
            raise ServiceError(f"{p['name']} can take at most {p['capacity_kg']:,.0f} kg.")
        if end_date and end_date < start_date:
            raise ServiceError("The end date is before the start date.")
        if p["kind"] == "transport" and not destination_market:
            raise ServiceError("Please choose where the produce is going.")
        group_key = None
        if p["kind"] == "transport" and share_transport:
            group_key = f"{provider_id}|{origin_market}|{destination_market}"
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO bookings (farmer_contact, provider_id, kind, crop, quantity_kg, "
                "start_date, end_date, origin_market, destination_market, share_transport, "
                "group_key, status, requested_at, advice_snapshot) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (farmer_contact, provider_id, p["kind"], crop, float(quantity_kg),
                 _iso(start_date), _iso(end_date), origin_market, destination_market,
                 int(bool(share_transport)), group_key, "requested", _iso(_now(now)),
                 json.dumps(advice_snapshot or {})))
            bid = int(cur.lastrowid)
            self._log(c, bid, "farmer", "requested", now=now)
            return bid

    def _move(self, booking_id: int, to: str, actor: str, *, note: str = "",
              condition: str = "", extra: dict | None = None, now: datetime | None = None) -> None:
        b = self.booking(booking_id)
        allowed = TRANSITIONS.get(b["status"], {})
        if to not in allowed:
            raise ServiceError(f"A booking that is '{STATUS_LABELS[b['status']]}' cannot be "
                               f"moved to '{STATUS_LABELS[to]}'.")
        if allowed[to] != actor:
            raise ServiceError(f"Only the {allowed[to]} can do that.")
        sets = {"status": to, **(extra or {})}
        if to in ("accepted", "quote_declined"):
            sets["decided_at"] = _iso(_now(now))
        with self._conn() as c:
            c.execute(f"UPDATE bookings SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?",
                      (*sets.values(), booking_id))
            self._log(c, booking_id, actor, to, note, condition, now)

    def _check_provider(self, booking_id: int, provider_id: int) -> dict:
        b = self.booking(booking_id)
        if b["provider_id"] != provider_id:
            raise ServiceError("This request belongs to another provider.")
        return b

    def _check_farmer(self, booking_id: int, farmer_contact: str) -> dict:
        b = self.booking(booking_id)
        if b["farmer_contact"] != farmer_contact:
            raise ServiceError("This booking belongs to another farmer.")
        return b

    def expire_overdue(self, now: datetime | None = None) -> list[int]:
        """Requests with no reply after QUOTE_DEADLINE_HOURS become 'expired'."""
        cutoff = _now(now) - timedelta(hours=cfg.QUOTE_DEADLINE_HOURS)
        old = self._df("SELECT id FROM bookings WHERE status = 'requested' AND requested_at < ?",
                       (_iso(cutoff),))["id"].tolist()
        for bid in old:
            self._move(int(bid), "expired", "system",
                       note=f"No quotation within {cfg.QUOTE_DEADLINE_HOURS} hours.", now=now)
        return [int(b) for b in old]

    def quote(self, booking_id: int, provider_id: int, amount_ugx: float, note: str = "",
              now: datetime | None = None) -> None:
        self._check_provider(booking_id, provider_id)
        if amount_ugx <= 0:
            raise ServiceError("The quotation must be greater than zero.")
        self._move(booking_id, "quoted", "provider", note=note,
                   extra={"quote_ugx": float(amount_ugx), "quote_note": note,
                          "quoted_at": _iso(_now(now))}, now=now)

    def decline_request(self, booking_id: int, provider_id: int, note: str = "",
                        now: datetime | None = None) -> None:
        self._check_provider(booking_id, provider_id)
        self._move(booking_id, "declined_by_provider", "provider", note=note, now=now)

    def accept_quote(self, booking_id: int, farmer_contact: str, now: datetime | None = None) -> None:
        self._check_farmer(booking_id, farmer_contact)
        self._move(booking_id, "accepted", "farmer", note="Quotation accepted.", now=now)

    def decline_quote(self, booking_id: int, farmer_contact: str, note: str = "",
                      now: datetime | None = None) -> None:
        self._check_farmer(booking_id, farmer_contact)
        self._move(booking_id, "quote_declined", "farmer", note=note, now=now)

    def cancel(self, booking_id: int, farmer_contact: str, note: str = "",
               now: datetime | None = None) -> None:
        self._check_farmer(booking_id, farmer_contact)
        self._move(booking_id, "cancelled", "farmer", note=note, now=now)

    def mark_received(self, booking_id: int, provider_id: int, condition: str, note: str = "",
                      now: datetime | None = None) -> None:
        if not condition.strip():
            raise ServiceError("Please record the condition of the produce.")
        self._check_provider(booking_id, provider_id)
        self._move(booking_id, "in_progress", "provider", note=note or "Produce received.",
                   condition=condition.strip(), now=now)

    def post_update(self, booking_id: int, provider_id: int, note: str, condition: str = "",
                    now: datetime | None = None) -> None:
        b = self._check_provider(booking_id, provider_id)
        if b["status"] not in ("accepted", "in_progress"):
            raise ServiceError("Updates can only be posted on confirmed or in-progress bookings.")
        if not note.strip():
            raise ServiceError("Please write the update.")
        with self._conn() as c:
            self._log(c, booking_id, "provider", None, note.strip(), condition.strip(), now)

    def complete(self, booking_id: int, provider_id: int, note: str = "",
                 now: datetime | None = None) -> None:
        self._check_provider(booking_id, provider_id)
        self._move(booking_id, "completed", "provider", note=note or "Completed.", now=now)

    def events(self, booking_id: int) -> pd.DataFrame:
        return self._df("SELECT * FROM booking_events WHERE booking_id = ? ORDER BY id",
                        (booking_id,))

    def _bookings_with_provider(self, where: str, params) -> pd.DataFrame:
        return self._df(
            "SELECT b.*, p.name AS provider_name, p.contact AS provider_contact, "
            "p.market AS provider_market, p.kind AS provider_kind, "
            "r.stars AS rating_stars, r.price_received_ugx_per_kg "
            "FROM bookings b JOIN providers p ON p.id = b.provider_id "
            "LEFT JOIN ratings r ON r.booking_id = b.id "
            f"WHERE {where} ORDER BY b.requested_at DESC", params)

    def farmer_bookings(self, farmer_contact: str) -> pd.DataFrame:
        return self._bookings_with_provider("b.farmer_contact = ?", (farmer_contact,))

    def provider_bookings(self, provider_id: int) -> pd.DataFrame:
        return self._bookings_with_provider("b.provider_id = ?", (provider_id,))

    def contacts_visible(self, booking: dict) -> bool:
        """Contacts are shared only once the farmer has accepted a quotation."""
        return booking["status"] in ("accepted", "in_progress", "completed")

    def next_ranked(self, booking_id: int, markets: pd.DataFrame,
                    transport_ugx_per_kg_km: float = cfg.TRANSPORT_UGX_PER_KG_KM) -> pd.DataFrame:
        """Other providers for an expired or declined request, best first."""
        b = self.booking(booking_id)
        tried = self._df("SELECT DISTINCT provider_id FROM bookings WHERE farmer_contact = ? "
                         "AND crop = ? AND kind = ?",
                         (b["farmer_contact"], b["crop"], b["kind"]))["provider_id"].tolist()
        months = 0
        if b["end_date"]:
            months = max(1, round((date.fromisoformat(b["end_date"]) -
                                   date.fromisoformat(b["start_date"])).days / 30))
        return self.rank_options(kind=b["kind"], crop=b["crop"], quantity_kg=b["quantity_kg"],
                                 home_market=b["origin_market"], markets=markets, months=months,
                                 transport_ugx_per_kg_km=transport_ugx_per_kg_km, exclude=tried)

    # ------------------------------------------------------------------ ratings
    def rate(self, booking_id: int, farmer_contact: str, stars: int, comment: str = "",
             price_received_ugx_per_kg: float | None = None, now: datetime | None = None) -> None:
        b = self._check_farmer(booking_id, farmer_contact)
        if b["status"] != "completed":
            raise ServiceError("You can rate a provider once the booking is completed.")
        if not 1 <= int(stars) <= 5:
            raise ServiceError("Choose between 1 and 5 stars.")
        try:
            with self._conn() as c:
                c.execute("INSERT INTO ratings (booking_id, provider_id, stars, comment, "
                          "price_received_ugx_per_kg, at) VALUES (?,?,?,?,?,?)",
                          (booking_id, b["provider_id"], int(stars), comment.strip(),
                           price_received_ugx_per_kg, _iso(_now(now))))
        except sqlite3.IntegrityError:
            raise ServiceError("You have already rated this booking.") from None

    # ------------------------------------------------------------------ shared transport
    def shared_groups(self, provider_id: int) -> list[list[int]]:
        """
        Open transport requests for this provider that opted in to sharing,
        grouped by route, with start dates within SHARED_TRIP_WINDOW_DAYS of
        the first request in the group. Groups of one are returned too.
        """
        b = self._df("SELECT id, group_key, start_date FROM bookings WHERE provider_id = ? "
                     "AND kind = 'transport' AND share_transport = 1 AND status = 'requested' "
                     "ORDER BY group_key, start_date", (provider_id,))
        groups = []
        for _, g in b.groupby("group_key"):
            current, first = [], None
            for _, r in g.iterrows():
                d = date.fromisoformat(r["start_date"])
                if first is None or (d - first).days > cfg.SHARED_TRIP_WINDOW_DAYS:
                    if current:
                        groups.append(current)
                    current, first = [], d
                current.append(int(r["id"]))
            if current:
                groups.append(current)
        return groups

    @staticmethod
    def split_trip_cost(price_per_trip: float, capacity_kg: float,
                        kgs: list[float]) -> tuple[int, list[float]]:
        """Trips needed for the group, and each farmer's share in proportion to kg."""
        total = sum(kgs)
        trips = max(1, math.ceil(total / capacity_kg))
        cost = price_per_trip * trips
        return trips, [round(cost * k / total) for k in kgs]

    def quote_group(self, booking_ids: list[int], provider_id: int, price_per_trip: float,
                    note: str = "", now: datetime | None = None) -> list[float]:
        """Quote one shared trip; each booking is quoted its kg share."""
        p = self.provider(provider_id)
        bookings = [self._check_provider(b, provider_id) for b in booking_ids]
        keys = {b["group_key"] for b in bookings}
        if len(keys) != 1 or None in keys:
            raise ServiceError("These requests are not on the same shared route.")
        trips, shares = self.split_trip_cost(price_per_trip, p["capacity_kg"],
                                             [b["quantity_kg"] for b in bookings])
        n = len(bookings)
        for b, share in zip(bookings, shares):
            msg = (f"Shared trip with {n - 1} other farmer(s): {trips} trip(s) at "
                   f"{price_per_trip:,.0f} UGX, your share by kg." if n > 1 else note)
            self.quote(b["id"], provider_id, share, msg if not note else f"{msg} {note}", now)
        return shares

    def sharing_estimate(self, provider_id: int, origin: str, destination: str,
                         start_date: date, quantity_kg: float) -> dict:
        """What a farmer would pay if they joined the open shared requests on this route."""
        p = self.provider(provider_id)
        key = f"{provider_id}|{origin}|{destination}"
        others = self._df("SELECT quantity_kg, start_date FROM bookings WHERE group_key = ? "
                          "AND status = 'requested'", (key,))
        if not others.empty:
            days = (pd.to_datetime(others["start_date"]).dt.date
                    .apply(lambda d: abs((d - start_date).days)))
            others = others[days <= cfg.SHARED_TRIP_WINDOW_DAYS]
        alone_trips, alone = self.split_trip_cost(p["price"], p["capacity_kg"], [quantity_kg])
        kgs = others["quantity_kg"].tolist() + [quantity_kg]
        trips, shares = self.split_trip_cost(p["price"], p["capacity_kg"], kgs)
        return {"others": int(len(others)), "others_kg": float(sum(kgs) - quantity_kg),
                "alone_ugx": alone[0], "shared_ugx": shares[-1], "trips": trips}
