"""
Add clearly fictional, verified demo providers so Part 2 can be shown end to end.

Run from the project root:

    python scripts/seed_demo_providers.py            # add demo providers
    python scripts/seed_demo_providers.py --remove   # delete them (and their bookings)

Every demo provider's name ends in "(demo)" and its contact is an
@example.com address, so it can never be mistaken for a real business.
Prices are rough placeholders, not quotes from real providers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src import data  # noqa: E402
from src.services import ServiceError, ServiceStore  # noqa: E402

GRAINS = ["Maize", "Beans", "Sorghum", "Millet"]

# (market, kind, price, capacity_kg, crops, description)
DEMO = []
for m in ["Lira", "Gulu", "Owino", "Mbarara", "Masaka", "Jinja", "Mbale", "Hoima", "Soroti",
          "Fort Portal", "Kasese", "Busia", "Iganga", "Arua", "Kotido"]:
    DEMO += [
        (m, "storage", 20, 50_000, GRAINS, "Warehouse with pallets and fumigation"),
    ]
    # one buyer per crop, so each pays a price that fits that crop
    DEMO += [(m, "buyer", 0, 20_000, [crop], f"Buys {crop.lower()}, pays on delivery")
             for crop in cfg.TARGET_CROPS]
for m in ["Lira", "Gulu", "Owino", "Mbarara", "Jinja", "Mbale", "Hoima", "Masaka"]:
    DEMO += [(m, "transport", 0, 5_000, [], "5-tonne lorry, local and long-distance trips")]
for m in ["Lira", "Owino", "Mbarara", "Gulu"]:
    DEMO += [(m, "storage", 25, 20_000, GRAINS, "Hermetic bags, smaller lots welcome"),
             (m, "preservation", 30, 30_000, GRAINS, "Drying and pest treatment before storage")]


def latest_price(panel, crop, market):
    s = panel[(panel["commodity"] == crop) & (panel["market"] == market) & panel["price_observed"].notna()]
    return float(s.sort_values("date")["price_observed"].iloc[-1]) if len(s) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    store = ServiceStore(cfg.SERVICES_DB)

    if args.remove:
        with store._conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM providers WHERE name LIKE '%(demo)'")]
            for pid in ids:
                bids = [r[0] for r in c.execute("SELECT id FROM bookings WHERE provider_id = ?", (pid,))]
                for b in bids:
                    c.execute("DELETE FROM booking_events WHERE booking_id = ?", (b,))
                    c.execute("DELETE FROM ratings WHERE booking_id = ?", (b,))
                c.execute("DELETE FROM bookings WHERE provider_id = ?", (pid,))
                c.execute("DELETE FROM providers WHERE id = ?", (pid,))
        print(f"Removed {len(ids)} demo providers from {cfg.SERVICES_DB}")
        return

    panel = data.load_monthly()
    markets = set(data.load_markets()["market"])
    added = skipped = 0
    counter: dict = {}
    for market, kind, price, capacity, crops, desc in DEMO:
        if market not in markets:
            continue
        counter[(market, kind)] = counter.get((market, kind), 0) + 1
        n = counter[(market, kind)]
        label = cfg.SERVICE_KINDS[kind]["label"].split(" (")[0]
        name = f"{market} {label} {n} (demo)"
        contact = f"demo-{kind}-{market.lower().replace(' ', '')}-{n}@example.com"
        if kind == "buyer":
            # pays about 85% of the latest retail price of that crop there, for illustration
            p = latest_price(panel, crops[0], market)
            if p is None:
                continue
            price = round(0.85 * p, -1)
            name = f"{market} {crops[0]} Buyer (demo)"
            contact = f"demo-buyer-{crops[0].lower().replace(' ', '')}-{market.lower().replace(' ', '')}@example.com"
        if kind == "transport":
            price = 350_000
        try:
            pid = store.register_provider(name=name, contact=contact, kind=kind, market=market,
                                          crops=crops, capacity_kg=capacity, price=price,
                                          description=desc, documents="Demo record, not a real business")
            store.set_provider_status(pid, "verified", "Demo provider")
            added += 1
        except ServiceError:
            skipped += 1
    print(f"Added {added} demo providers ({skipped} already there) to {cfg.SERVICES_DB}")


if __name__ == "__main__":
    main()
