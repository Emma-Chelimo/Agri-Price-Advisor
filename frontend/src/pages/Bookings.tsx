import { useCallback, useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, dateTime, dayMonth, num, perKg, ugx, type Booking } from "../api";
import { useApp } from "../state";

const ACTIVE = ["requested", "quoted", "accepted", "in_progress"];

export function pillClass(status: string) {
  if (status === "quoted") return "pill wait";
  if (status === "completed") return "pill done";
  if (["declined_by_provider", "quote_declined", "expired", "cancelled"].includes(status)) return "pill bad";
  return "pill";
}

export function contactLink(c: string) {
  return c.includes("@") ? `mailto:${c}` : `tel:${c}`;
}

export default function Bookings() {
  const [items, setItems] = useState<Booking[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(() => {
    api<Booking[]>("/api/bookings").then(setItems).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  async function act(b: Booking, action: string, body: unknown = {}, done?: string) {
    setError(null); setMsg(null);
    try {
      const r = await api<{ provider_name?: string }>(`/api/bookings/${b.id}/${action}`, { body });
      setMsg(done ?? (r.provider_name ? `Request sent to ${r.provider_name}.` : null));
      load();
    } catch (e) { setError((e as Error).message); }
  }

  if (!items) return <div className="page">{error ? <p className="notice warn">{error}</p> : <p className="faint">Loading your bookings…</p>}</div>;

  const active = items.filter((b) => ACTIVE.includes(b.status));
  const past = items.filter((b) => !ACTIVE.includes(b.status));

  return (
    <div className="page narrow stack">
      <div>
        <h1>My bookings</h1>
        <p className="muted">Requests you've sent to storage, transport and buyers, and where each one stands.</p>
      </div>
      {msg && <p className="notice" role="status">{msg}</p>}
      {error && <p className="notice warn" role="alert">{error}</p>}

      {items.length === 0 ? (
        <div className="empty">
          <p>You haven't sent any requests yet. Get advice for your crop first, then find storage, transport or a buyer from there.</p>
          <Link className="btn primary" to="/sell">Plan a sale</Link>
        </div>
      ) : (
        <>
          <section>
            <h2>Open</h2>
            {active.length === 0 ? <p className="faint">Nothing open right now.</p> : (
              <ul className="list">{active.map((b) => <Item key={b.id} b={b} act={act} />)}</ul>
            )}
          </section>
          {past.length > 0 && (
            <section className="section">
              <h2>Finished</h2>
              <ul className="list">{past.map((b) => <Item key={b.id} b={b} act={act} />)}</ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}

type Act = (b: Booking, action: string, body?: unknown, done?: string) => Promise<void>;

function Item({ b, act }: { b: Booking; act: Act }) {
  const { meta } = useApp();
  const label = meta?.service_kinds[b.kind]?.label ?? b.kind;
  const route = b.destination_market ? `${b.origin_market} to ${b.destination_market}` : b.origin_market;
  const when = b.end_date ? `${dayMonth(b.start_date)} to ${dayMonth(b.end_date)}` : dayMonth(b.start_date);

  return (
    <li>
      <div className="list-row">
        <div>
          <strong style={{ fontSize: "var(--fs-md)" }}>{label}: {b.provider_name}</strong>
          <div className="faint">{num(b.quantity_kg)} kg of {b.crop.toLowerCase()}, {route}, {when}{b.share_transport ? ", shared trip" : ""}.</div>
        </div>
        <span className={pillClass(b.status)}>{b.status_label}</span>
      </div>

      {b.status === "requested" && (
        <p className="faint" style={{ marginTop: 8 }}>
          Sent {dateTime(b.requested_at)}. If {b.provider_name} doesn't reply within {meta?.quote_deadline_hours ?? 48} hours we'll offer you the next provider.
        </p>
      )}

      {b.status === "quoted" && (
        <div className="panel" style={{ marginTop: 12 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span>Quotation</span><span className="figure big">{ugx(b.quote_ugx)}</span>
          </div>
          {b.quote_note && <p className="muted" style={{ marginTop: 8 }}>{b.quote_note}</p>}
          <p className="faint">Accepting shares your contact with {b.provider_name} and shows you theirs.</p>
          <div className="row">
            <button className="btn primary" onClick={() => act(b, "accept", {}, `You accepted the quotation from ${b.provider_name}.`)}>Accept quotation</button>
            <button className="btn danger" onClick={() => act(b, "decline", {}, "Quotation declined.")}>Decline</button>
          </div>
        </div>
      )}

      {b.contacts_visible && b.provider_contact && (
        <div className="notice" style={{ marginTop: 12 }}>
          Contact {b.provider_name}: <a href={contactLink(b.provider_contact)}><strong>{b.provider_contact}</strong></a>
          {b.map_link && <div><a href={b.map_link} target="_blank" rel="noreferrer">Directions to {b.provider_market}</a></div>}
          {b.quote_ugx != null && <div className="faint">Agreed price: {ugx(b.quote_ugx)}</div>}
        </div>
      )}

      {b.recheck && (
        <div className="notice gold" style={{ marginTop: 12 }}>
          Your storage ends soon. With prices up to {dayMonth(b.recheck.date)} ({perKg(b.recheck.price)}), the advice for {b.origin_market} is now <strong>{b.recheck.recommendation.toLowerCase()}</strong>.
        </div>
      )}

      {(b.status === "expired" || b.status === "declined_by_provider") && (
        b.next_option ? (
          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn small primary" onClick={() => act(b, "resend")}>Send to {b.next_option.name} instead</button>
            <span className="faint">{b.next_option.market}</span>
          </div>
        ) : <p className="faint" style={{ marginTop: 8 }}>No other verified provider is available right now.</p>
      )}

      {b.status === "completed" && b.rating_stars == null && <RateForm b={b} act={act} />}
      {b.rating_stars != null && <p className="faint" style={{ marginTop: 8 }}>You rated this {b.rating_stars} of 5.</p>}

      <div className="row" style={{ marginTop: 8 }}>
        <details>
          <summary>History</summary>
          <ol className="timeline">
            {b.events.map((e, i) => (
              <li key={i}>
                <strong>{e.label}</strong> <span className="faint">{dateTime(e.at)}</span>
                {e.condition && <div>Condition: {e.condition}</div>}
                {e.note && <div className="muted">{e.note}</div>}
              </li>
            ))}
          </ol>
        </details>
        {["requested", "quoted", "accepted"].includes(b.status) && (
          <button className="link" style={{ color: "var(--maroon)", marginLeft: "auto" }}
            onClick={() => { if (window.confirm(`Cancel this ${label.toLowerCase()} request?`)) act(b, "cancel", {}, "Booking cancelled."); }}>
            Cancel booking
          </button>
        )}
      </div>
    </li>
  );
}

function RateForm({ b, act }: { b: Booking; act: Act }) {
  const [stars, setStars] = useState(0);
  const [price, setPrice] = useState("");
  const [comment, setComment] = useState("");
  function submit(e: FormEvent) {
    e.preventDefault();
    if (!stars) return;
    act(b, "rate", { stars, comment, price_received_ugx_per_kg: price ? Number(price) : null }, "Thanks, your rating helps other farmers choose.");
  }
  return (
    <form className="panel stack" style={{ marginTop: 12 }} onSubmit={submit}>
      <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 700, marginBottom: 8 }}>How was {b.provider_name}?</legend>
        <div className="chips">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} type="button" className="chip" aria-pressed={stars === n} onClick={() => setStars(n)}>
              {n} star{n > 1 ? "s" : ""}
            </button>
          ))}
        </div>
      </fieldset>
      {(b.kind === "buyer" || b.kind === "transport") && (
        <label className="field">
          <span>Price you were paid, UGX per kg <span className="faint">(optional)</span></span>
          <input type="number" min={0} value={price} onChange={(e) => setPrice(e.target.value)} />
          <small>This helps us check how good our advice was.</small>
        </label>
      )}
      <label className="field">
        <span>Comment <span className="faint">(optional)</span></span>
        <input type="text" maxLength={300} value={comment} onChange={(e) => setComment(e.target.value)} />
      </label>
      <button className="btn primary" disabled={!stars}>Send rating</button>
    </form>
  );
}
