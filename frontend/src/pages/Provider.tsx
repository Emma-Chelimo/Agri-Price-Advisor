import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, dateTime, dayMonth, num, ugx, type Booking } from "../api";
import { useApp } from "../state";
import { contactLink, pillClass } from "./Bookings";

interface Group { booking_ids: number[]; total_kg: number; origin: string; destination: string; from: string; to: string; suggested_trips: number; suggested_shares: number[] }
interface Service {
  id: number; name: string; kind: string; market: string; crops: string; capacity_kg: number; price: number;
  description: string; status: "pending" | "verified" | "rejected"; admin_note: string; created_at: string;
  rating: number | null; rating_count: number; bookings: Booking[]; shared_groups: Group[];
}

const STATUS_TEXT = {
  pending: "Waiting for approval",
  verified: "Approved",
  rejected: "Not approved",
};

export default function Provider() {
  const [services, setServices] = useState<Service[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  const load = useCallback(() => {
    api<{ services: Service[] }>("/api/provider/me").then((r) => setServices(r.services)).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function act(path: string, body: unknown, done: string) {
    setError(null); setMsg(null);
    try { await api(path, { body }); setMsg(done); load(); }
    catch (e) { setError((e as Error).message); }
  }

  if (!services) return <div className="page">{error ? <p className="notice warn">{error}</p> : <p className="faint">Loading…</p>}</div>;

  if (services.length === 0 || adding) {
    return (
      <div className="page narrow stack">
        <div>
          <h1>{services.length === 0 ? "Offer your service to farmers" : "Register another service"}</h1>
          <p className="muted">
            Tell us what you offer. An administrator checks every new service before farmers can see it, usually within two working days.
          </p>
        </div>
        <Register onDone={() => { setAdding(false); setMsg("Thanks. Your service is waiting for approval."); load(); }}
          onCancel={services.length ? () => setAdding(false) : undefined} />
      </div>
    );
  }

  const waiting = services.reduce((n, s) => n + s.bookings.filter((b) => b.status === "requested").length, 0);

  return (
    <div className="page narrow stack">
      <div>
        <h1>Your services</h1>
        <p className="muted">
          {waiting ? `${waiting} request${waiting > 1 ? "s need" : " needs"} a quotation from you.` : "No requests are waiting for a reply."}
        </p>
      </div>
      {msg && <p className="notice" role="status">{msg}</p>}
      {error && <p className="notice warn" role="alert">{error}</p>}
      {services.map((s) => <ServiceBlock key={s.id} s={s} act={act} />)}
      <button className="btn" onClick={() => setAdding(true)}>Register another service</button>
    </div>
  );
}

type Act = (path: string, body: unknown, done: string) => Promise<void>;

function ServiceBlock({ s, act }: { s: Service; act: Act }) {
  const { meta } = useApp();
  const kind = meta?.service_kinds[s.kind];
  const open = s.bookings.filter((b) => ["requested", "quoted", "accepted", "in_progress"].includes(b.status));
  const closed = s.bookings.filter((b) => !open.includes(b));
  const order = ["requested", "accepted", "in_progress", "quoted"];
  open.sort((a, b) => order.indexOf(a.status) - order.indexOf(b.status));

  return (
    <section className="section">
      <div className="list-row">
        <div>
          <h2 style={{ marginBottom: 4 }}>{s.name}</h2>
          <div className="faint">
            {kind?.label ?? s.kind} in {s.market}. {num(s.price)} {kind?.price_unit}, up to {num(s.capacity_kg)} kg.
            {s.rating != null && ` Rated ${s.rating.toFixed(1)} of 5 by ${s.rating_count}.`}
          </div>
        </div>
        <span className={s.status === "verified" ? "pill" : s.status === "pending" ? "pill wait" : "pill bad"}>{STATUS_TEXT[s.status]}</span>
      </div>
      {s.status !== "verified" && (
        <p className="faint" style={{ marginTop: 8 }}>
          {s.status === "pending" ? "Farmers will see this service once an administrator approves it." : `Reason: ${s.admin_note || "none given"}.`}
        </p>
      )}

      {s.shared_groups.map((g) => <SharedGroup key={g.booking_ids.join("-")} g={g} price={s.price} act={act} />)}

      {s.status === "verified" && (open.length === 0
        ? <p className="faint" style={{ marginTop: 12 }}>No open requests.</p>
        : <ul className="list" style={{ marginTop: 12 }}>{open.map((b) => <Request key={b.id} b={b} kind={s.kind} act={act} />)}</ul>)}

      {closed.length > 0 && (
        <details style={{ marginTop: 8 }}>
          <summary>{closed.length} finished request{closed.length > 1 ? "s" : ""}</summary>
          <ul className="list">
            {closed.map((b) => (
              <li key={b.id} className="list-row">
                <span>{num(b.quantity_kg)} kg of {b.crop.toLowerCase()}, {dayMonth(b.start_date)}{b.rating_stars ? `, rated ${b.rating_stars} of 5` : ""}</span>
                <span className={pillClass(b.status)}>{b.status_label}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function Request({ b, kind, act }: { b: Booking; kind: string; act: Act }) {
  const [amount, setAmount] = useState("");
  const [note, setNote] = useState("");
  const route = b.destination_market ? `${b.origin_market} to ${b.destination_market}` : `from ${b.origin_market}`;
  const when = b.end_date ? `${dayMonth(b.start_date)} to ${dayMonth(b.end_date)}` : dayMonth(b.start_date);
  const base = `/api/provider/bookings/${b.id}`;

  function submit(e: FormEvent, path: string, body: unknown, done: string) {
    e.preventDefault();
    act(`${base}/${path}`, body, done).then(() => { setAmount(""); setNote(""); });
  }

  return (
    <li>
      <div className="list-row">
        <div>
          <strong>{num(b.quantity_kg)} kg of {b.crop.toLowerCase()}</strong>
          <div className="faint">{route}, {when}{b.share_transport ? ", happy to share a trip" : ""}.</div>
        </div>
        <span className={pillClass(b.status)}>{b.status_label}</span>
      </div>

      {b.status === "requested" && (
        <form className="panel stack" style={{ marginTop: 12 }} onSubmit={(e) => submit(e, "quote", { amount_ugx: Number(amount), note }, "Quotation sent.")}>
          {b.reply_by && <p className="faint" style={{ margin: 0 }}>Reply by {dateTime(b.reply_by)}, or the farmer is offered another provider.</p>}
          <div className="row" style={{ alignItems: "flex-end" }}>
            <label className="field" style={{ flex: "1 1 160px" }}>
              <span>Your price for this job, UGX</span>
              <input type="number" min={1} required value={amount} onChange={(e) => setAmount(e.target.value)} />
            </label>
            <label className="field" style={{ flex: "2 1 220px" }}>
              <span>Note <span className="faint">(optional)</span></span>
              <input type="text" maxLength={300} value={note} onChange={(e) => setNote(e.target.value)}
                placeholder={kind === "storage" ? "For example: bring dry, bagged grain" : ""} />
            </label>
          </div>
          <div className="row">
            <button className="btn primary" disabled={!(Number(amount) > 0)}>Send quotation</button>
            <button type="button" className="link" style={{ color: "var(--maroon)" }}
              onClick={() => act(`${base}/decline`, { note }, "Request declined.")}>I can't take this job</button>
          </div>
        </form>
      )}

      {b.status === "quoted" && <p className="faint" style={{ marginTop: 8 }}>You quoted {ugx(b.quote_ugx)}. Waiting for the farmer to accept.</p>}

      {b.contacts_visible && b.farmer_contact && (
        <p className="notice" style={{ marginTop: 12 }}>
          Farmer's contact: <a href={contactLink(b.farmer_contact)}><strong>{b.farmer_contact}</strong></a>. Agreed price {ugx(b.quote_ugx)}.
        </p>
      )}

      {b.status === "accepted" && (
        <form className="row" style={{ marginTop: 12, alignItems: "flex-end" }}
          onSubmit={(e) => submit(e, "receive", { condition: note }, "Marked as received.")}>
          <label className="field" style={{ flex: "1 1 240px" }}>
            <span>{kind === "transport" ? "Condition when loaded" : "Condition when received"}</span>
            <input type="text" required maxLength={300} placeholder="For example: dry, 10 bags" value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          <button className="btn" disabled={!note.trim()}>{kind === "transport" ? "Mark as collected" : "Mark as received"}</button>
        </form>
      )}

      {b.status === "in_progress" && (
        <div className="stack" style={{ marginTop: 12 }}>
          <form className="row" style={{ alignItems: "flex-end" }} onSubmit={(e) => submit(e, "update", { note }, "Update sent to the farmer.")}>
            <label className="field" style={{ flex: "1 1 240px" }}>
              <span>Send the farmer an update</span>
              <input type="text" required maxLength={300} value={note} onChange={(e) => setNote(e.target.value)} />
            </label>
            <button className="btn" disabled={!note.trim()}>Send update</button>
          </form>
          <button className="btn primary" onClick={() => act(`${base}/complete`, {}, "Job marked as complete.")}>Mark job complete</button>
        </div>
      )}
    </li>
  );
}

function SharedGroup({ g, price, act }: { g: Group; price: number; act: Act }) {
  const [trip, setTrip] = useState(String(price));
  return (
    <div className="notice gold" style={{ marginTop: 12 }}>
      <strong>{g.booking_ids.length} farmers can share one route</strong>: {g.origin} to {g.destination}, {dayMonth(g.from)} to {dayMonth(g.to)}, {num(g.total_kg)} kg in all.
      At {ugx(Number(trip) || price)} a trip that's {g.suggested_trips} trip{g.suggested_trips > 1 ? "s" : ""}, split by kg.
      <form className="row" style={{ marginTop: 8 }}
        onSubmit={(e) => { e.preventDefault(); act("/api/provider/groups/quote", { booking_ids: g.booking_ids, price_per_trip: Number(trip) }, "Shared quotation sent to every farmer in the group."); }}>
        <input type="number" min={1} aria-label="Price per trip, UGX" style={{ width: 180 }} value={trip} onChange={(e) => setTrip(e.target.value)} />
        <button className="btn small primary" disabled={!(Number(trip) > 0)}>Quote the group</button>
      </form>
    </div>
  );
}

function Register({ onDone, onCancel }: { onDone: () => void; onCancel?: () => void }) {
  const { meta } = useApp();
  const [f, setF] = useState({ name: "", kind: "storage", market: "", crops: [] as string[], capacity_kg: "", price: "", description: "", documents: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (p: Partial<typeof f>) => setF((x) => ({ ...x, ...p }));
  if (!meta) return null;
  const unit = meta.service_kinds[f.kind]?.price_unit;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      await api("/api/provider/register", { body: { ...f, capacity_kg: Number(f.capacity_kg), price: Number(f.price) } });
      onDone();
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }

  return (
    <form className="stack" onSubmit={submit}>
      <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 700, marginBottom: 8 }}>What do you offer?</legend>
        <div className="choices">
          {Object.entries(meta.service_kinds).map(([k, v]) => (
            <label key={k} className="choice">
              <input type="radio" name="kind" checked={f.kind === k} onChange={() => set({ kind: k })} />{v.label}
            </label>
          ))}
        </div>
      </fieldset>
      <label className="field"><span>Business name</span>
        <input type="text" required maxLength={100} value={f.name} onChange={(e) => set({ name: e.target.value })} /></label>
      <label className="field"><span>Market you work from</span>
        <select required value={f.market} onChange={(e) => set({ market: e.target.value })}>
          <option value="">Choose a market</option>
          {meta.all_markets.map((m) => <option key={m}>{m}</option>)}
        </select></label>
      <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 700, marginBottom: 8 }}>Crops you handle <span className="faint">(none ticked means all)</span></legend>
        <div className="chips">
          {meta.crops.map((c) => (
            <button key={c} type="button" className="chip" aria-pressed={f.crops.includes(c)}
              onClick={() => set({ crops: f.crops.includes(c) ? f.crops.filter((x) => x !== c) : [...f.crops, c] })}>{c}</button>
          ))}
        </div>
      </fieldset>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <label className="field" style={{ flex: "1 1 180px" }}><span>{f.kind === "transport" ? "Truck load, kg" : f.kind === "buyer" ? "Most you buy at once, kg" : "Room for, kg"}</span>
          <input type="number" min={1} required value={f.capacity_kg} onChange={(e) => set({ capacity_kg: e.target.value })} /></label>
        <label className="field" style={{ flex: "1 1 180px" }}><span>Price, {unit}</span>
          <input type="number" min={1} required value={f.price} onChange={(e) => set({ price: e.target.value })} /></label>
      </div>
      <label className="field"><span>Short description <span className="faint">(optional)</span></span>
        <textarea maxLength={300} value={f.description} onChange={(e) => set({ description: e.target.value })} /></label>
      <label className="field"><span>Licence or registration number <span className="faint">(optional)</span></span>
        <input type="text" maxLength={300} value={f.documents} onChange={(e) => set({ documents: e.target.value })} />
        <small>Helps the administrator approve you faster.</small></label>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="row">
        <button className="btn primary" disabled={busy}>{busy ? "Sending…" : "Send for approval"}</button>
        {onCancel && <button type="button" className="link" onClick={onCancel}>Cancel</button>}
      </div>
    </form>
  );
}
