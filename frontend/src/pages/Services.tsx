import { useEffect, useState, type FormEvent } from "react";
import { Link, Navigate } from "react-router-dom";
import { api, num, ugx, type Provider } from "../api";
import { adviceRequest, useApp } from "../state";

const SHOWN = 3;

interface Group { kind: string; heading: string; providers: Provider[] }
interface Options { recommendation: string; months: number; market: string; sell_market: string; groups: Group[]; other_kinds: string[] }

function isoIn(days: number, from = new Date()) {
  const d = new Date(from); d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

export default function Services() {
  const { plan, meta } = useApp();
  const req = adviceRequest(plan);
  const [opts, setOpts] = useState<Options | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [extra, setExtra] = useState<Group[]>([]);
  const [sent, setSent] = useState<string | null>(null);

  useEffect(() => {
    if (!req) return;
    api<Options>("/api/services/options", { body: req }).then(setOpts).catch((e) => setError(e.message));
  }, [JSON.stringify(req)]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!req) return <Navigate to="/sell" replace />;
  if (error) return <div className="page narrow"><p className="notice warn">{error}</p></div>;
  if (!opts || !meta) return <div className="page"><p className="faint">Finding services near you…</p></div>;

  async function addKind(kind: string) {
    if (!req || extra.some((g) => g.kind === kind)) return;
    const q = new URLSearchParams({ crop: req.crop, quantity_kg: String(req.quantity_kg), market: req.market, months: String(Math.max(opts!.months, 1)) });
    try {
      const providers = await api<Provider[]>(`/api/services/${kind}?${q}`);
      setExtra((x) => [...x, { kind, heading: meta!.service_kinds[kind].label, providers }]);
    } catch (e) { setError((e as Error).message); }
  }

  const groups = [...opts.groups, ...extra];
  const others = opts.other_kinds.filter((k) => !extra.some((g) => g.kind === k));

  return (
    <div className="page narrow stack">
      <div>
        <p style={{ marginBottom: 4 }}><Link to="/advice">Back to my advice</Link></p>
        <h1>{opts.months > 0 ? "Find storage" : opts.sell_market !== opts.market ? `Get your ${req.crop.toLowerCase()} to ${opts.sell_market}` : "Find a buyer"}</h1>
        <p className="muted">
          Verified providers who take {req.crop.toLowerCase()} and have room for {num(req.quantity_kg)} kg, cheapest total first.
          Sending a request is free. The provider has {meta.quote_deadline_hours} hours to send you a quotation, and you only
          share your contact once you accept it.
        </p>
      </div>

      {sent && (
        <div className="notice" role="status">
          {sent} <Link to="/bookings">Follow it in My bookings</Link>.
        </div>
      )}

      {groups.map((g) => (
        <ProviderGroup key={g.kind} group={g} unit={meta.service_kinds[g.kind]?.price_unit ?? ""} opts={opts}
          onSent={(name) => { setSent(`Request sent to ${name}.`); window.scrollTo({ top: 0, behavior: "smooth" }); }} />
      ))}

      {others.length > 0 && (
        <section className="section">
          <h2 style={{ fontSize: "var(--fs-md)" }}>Need something else?</h2>
          <div className="chips">
            {others.map((k) => (
              <button key={k} className="chip" onClick={() => addKind(k)}>{meta.service_kinds[k].label}</button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function ProviderGroup({ group, unit, opts, onSent }: { group: Group; unit: string; opts: Options; onSent: (name: string) => void }) {
  const [all, setAll] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const list = all ? group.providers : group.providers.slice(0, SHOWN);
  return (
    <section className="section">
      <h2>{group.heading}</h2>
      {group.providers.length === 0 ? (
        <p className="empty" style={{ padding: "8px 0" }}>
          No verified {group.heading.toLowerCase().startsWith("buyers") ? "buyer" : "provider"} matches yet. Ask a provider you know to
          register on AgriPrice Advisor, or check back in a few days.
        </p>
      ) : (
        <ul className="list">
          {list.map((p, i) => (
            <li key={p.id}>
              <div className="list-row">
                <div>
                  {i === 0 && <div><span className="pill">{group.kind === "buyer" ? "Pays most" : "Lowest cost"}</span></div>}
                  <strong style={{ fontSize: "var(--fs-md)" }}>{p.name}</strong>
                  <div className="faint">
                    {p.market}{p.distance_km > 0 ? `, ${num(p.distance_km)} km away` : ", in your market"}.{" "}
                    {p.rating != null ? `Rated ${p.rating.toFixed(1)} of 5 by ${p.rating_count} farmer${p.rating_count === 1 ? "" : "s"}.` : "No ratings yet."}
                  </div>
                  {p.description && <div className="faint">{p.description}</div>}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div className="faint">{group.kind === "buyer" ? "You receive about" : "About"}</div>
                  <div className="figure">{ugx(p.estimate_ugx)}</div>
                  <div className="faint">{num(p.price)} {unit}{p.trips ? `, ${p.trips} trip${p.trips > 1 ? "s" : ""}` : ""}</div>
                </div>
              </div>
              {open === p.id
                ? <RequestForm p={p} kind={group.kind} opts={opts} onCancel={() => setOpen(null)} onSent={() => { setOpen(null); onSent(p.name); }} />
                : <button className="btn small" style={{ marginTop: 12 }} onClick={() => setOpen(p.id)}>Request a quotation</button>}
            </li>
          ))}
        </ul>
      )}
      {group.providers.length > SHOWN && (
        <button className="link" style={{ marginTop: 12 }} onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${group.providers.length}`}
        </button>
      )}
    </section>
  );
}

interface ShareEstimate { others: number; others_kg: number; alone_ugx: number; shared_ugx: number; trips: number }

function RequestForm({ p, kind, opts, onCancel, onSent }: { p: Provider; kind: string; opts: Options; onCancel: () => void; onSent: () => void }) {
  const { plan, meta } = useApp();
  const months = Math.max(opts.months, 1);
  const [start, setStart] = useState(isoIn(kind === "transport" ? 2 : 1));
  const [end, setEnd] = useState(kind === "storage" ? isoIn(months * 30, new Date(isoIn(1))) : "");
  const [dest, setDest] = useState(kind === "transport" ? opts.sell_market : "");
  const [share, setShare] = useState(false);
  const [est, setEst] = useState<ShareEstimate | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (kind !== "transport" || !dest || !plan.market) return;
    const q = new URLSearchParams({ provider_id: String(p.id), origin: plan.market, destination: dest, start_date: start, quantity_kg: String(plan.quantity_kg) });
    api<ShareEstimate>(`/api/transport/sharing-estimate?${q}`).then(setEst).catch(() => setEst(null));
  }, [kind, dest, start, p.id, plan.market, plan.quantity_kg]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      await api("/api/bookings", { body: {
        provider_id: p.id, crop: plan.crop, quantity_kg: plan.quantity_kg, start_date: start,
        end_date: kind === "storage" ? end : null, origin_market: plan.market,
        destination_market: kind === "transport" ? dest : null, share_transport: kind === "transport" && share,
        advice_snapshot: { recommendation: opts.recommendation, months: opts.months, sell_market: opts.sell_market },
      } });
      onSent();
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }

  return (
    <form className="panel stack" style={{ marginTop: 12 }} onSubmit={submit}>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <label className="field" style={{ flex: "1 1 180px" }}>
          <span>{kind === "storage" ? "Bring the crop in on" : kind === "transport" ? "Collect on" : kind === "buyer" ? "Ready to sell on" : "Start on"}</span>
          <input type="date" min={isoIn(0)} value={start} onChange={(e) => setStart(e.target.value)} required />
        </label>
        {kind === "storage" && (
          <label className="field" style={{ flex: "1 1 180px" }}>
            <span>Take it out on</span>
            <input type="date" min={start} value={end} onChange={(e) => setEnd(e.target.value)} required />
          </label>
        )}
        {kind === "transport" && (
          <label className="field" style={{ flex: "1 1 180px" }}>
            <span>Deliver to</span>
            <select value={dest} onChange={(e) => setDest(e.target.value)} required>
              <option value="">Choose a market</option>
              {(meta?.all_markets ?? []).filter((m) => m !== plan.market).map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
        )}
      </div>
      {kind === "transport" && (
        <label className="check">
          <input type="checkbox" checked={share} onChange={(e) => setShare(e.target.checked)} />
          <span>
            Share the truck with other farmers going the same way within a few days, and split the cost by kg.
            {est && (est.others > 0
              ? <> Right now {est.others} other farmer{est.others > 1 ? "s are" : " is"} sending {num(est.others_kg)} kg: your share would be about <strong>{ugx(est.shared_ugx)}</strong> instead of {ugx(est.alone_ugx)}.</>
              : <> Nobody else has asked for this route yet. Alone it costs about {ugx(est.alone_ugx)}.</>)}
          </span>
        </label>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="row">
        <button className="btn primary" disabled={busy}>{busy ? "Sending…" : `Send request to ${p.name}`}</button>
        <button type="button" className="link" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}
