import { Fragment, useCallback, useEffect, useState } from "react";
import { api, dayMonth, num } from "../api";
import { useApp } from "../state";

interface P {
  id: number; name: string; contact: string; kind: string; market: string; crops: string; capacity_kg: number; price: number;
  description: string; documents: string; status: string; admin_note: string; created_at: string;
  rating: number | null; rating_count: number;
}
interface Overview { bookings: Record<string, number>; providers: Record<string, number> }

const TABS = [["pending", "Waiting for approval"], ["verified", "Approved"], ["rejected", "Not approved"]] as const;

export default function Admin() {
  const { meta } = useApp();
  const [tab, setTab] = useState<string>("pending");
  const [list, setList] = useState<P[] | null>(null);
  const [ov, setOv] = useState<Overview | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api<P[]>(`/api/admin/providers?status=${tab}`).then(setList).catch((e) => setError(e.message));
    api<Overview>("/api/admin/overview").then(setOv).catch(() => {});
  }, [tab]);
  useEffect(load, [load]);

  async function setStatus(p: P, status: string, note = "") {
    setError(null);
    try {
      await api(`/api/admin/providers/${p.id}/status`, { body: { status, note } });
      setMsg(`${p.name} is now ${status === "verified" ? "approved" : status === "rejected" ? "not approved" : "waiting for approval"}.`);
      load();
    } catch (e) { setError((e as Error).message); }
  }

  async function reload() {
    setError(null);
    try {
      const r = await api<{ series: number }>("/api/admin/reload-data", { body: {} });
      setMsg(`Price data reloaded: ${r.series} market and crop series.`);
    } catch (e) { setError((e as Error).message); }
  }

  const bookingTotal = ov ? Object.values(ov.bookings).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="page stack">
      <h1>Administration</h1>
      {msg && <p className="notice" role="status">{msg}</p>}
      {error && <p className="notice warn" role="alert">{error}</p>}

      {ov && (
        <section className="split" style={{ gap: 24 }}>
          <div className="panel">
            <h2 style={{ fontSize: "var(--fs-md)" }}>Providers</h2>
            <dl style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "4px 16px", margin: 0 }}>
              {TABS.map(([k, label]) => <Fragment key={k}><dt>{label}</dt><dd className="figure" style={{ margin: 0, textAlign: "right" }}>{ov.providers[k] ?? 0}</dd></Fragment>)}
            </dl>
          </div>
          <div className="panel">
            <h2 style={{ fontSize: "var(--fs-md)" }}>Bookings ({bookingTotal})</h2>
            {bookingTotal === 0 ? <p className="faint">No bookings yet.</p> : (
              <dl style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "4px 16px", margin: 0 }}>
                {Object.entries(ov.bookings).map(([k, n]) => <Fragment key={k}><dt>{k}</dt><dd className="figure" style={{ margin: 0, textAlign: "right" }}>{n}</dd></Fragment>)}
              </dl>
            )}
          </div>
        </section>
      )}

      <section className="section">
        <div className="chips" role="tablist" aria-label="Provider status">
          {TABS.map(([k, label]) => (
            <button key={k} role="tab" className="chip" aria-selected={tab === k} aria-pressed={tab === k} onClick={() => setTab(k)}>
              {label}{ov?.providers[k] ? ` (${ov.providers[k]})` : ""}
            </button>
          ))}
        </div>
        {!list ? <p className="faint">Loading…</p> : list.length === 0 ? (
          <p className="empty">{tab === "pending" ? "Nothing to review. New registrations appear here." : "None."}</p>
        ) : (
          <ul className="list" style={{ marginTop: 16 }}>
            {list.map((p) => <ProviderRow key={p.id} p={p} unit={meta?.service_kinds[p.kind]?.price_unit ?? ""} label={meta?.service_kinds[p.kind]?.label ?? p.kind} setStatus={setStatus} />)}
          </ul>
        )}
      </section>

      <section className="section">
        <h2 style={{ fontSize: "var(--fs-md)" }}>Price data</h2>
        <p className="muted">After rebuilding the dataset or choosing new models, reload so the advice uses them without restarting the server.</p>
        <button className="btn" onClick={reload}>Reload price data</button>
      </section>
    </div>
  );
}

function ProviderRow({ p, unit, label, setStatus }: { p: P; unit: string; label: string; setStatus: (p: P, s: string, note?: string) => void }) {
  const [note, setNote] = useState("");
  return (
    <li>
      <div className="list-row">
        <div>
          <strong style={{ fontSize: "var(--fs-md)" }}>{p.name}</strong>
          <div className="faint">{label} in {p.market}. {num(p.price)} {unit}, up to {num(p.capacity_kg)} kg. Crops: {p.crops || "all"}.</div>
          <div className="faint">Contact {p.contact}. Registered {dayMonth(p.created_at)}.{p.rating != null && ` Rated ${p.rating.toFixed(1)} of 5 by ${p.rating_count}.`}</div>
          {p.description && <div className="muted">{p.description}</div>}
          {p.documents && <div>Licence or registration: <strong>{p.documents}</strong></div>}
          {p.admin_note && <div className="faint">Note: {p.admin_note}</div>}
        </div>
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        {p.status !== "verified" && <button className="btn small primary" onClick={() => setStatus(p, "verified")}>Approve</button>}
        {p.status !== "rejected" && (
          <>
            <input type="text" aria-label="Reason for not approving" placeholder="Reason, shown to the provider" style={{ flex: "1 1 200px", width: "auto", minHeight: 38 }}
              value={note} onChange={(e) => setNote(e.target.value)} />
            <button className="btn small danger" disabled={!note.trim()} onClick={() => setStatus(p, "rejected", note)}>
              {p.status === "verified" ? "Withdraw approval" : "Don't approve"}
            </button>
          </>
        )}
        {p.status === "rejected" && <button className="btn small" onClick={() => setStatus(p, "pending")}>Review again</button>}
      </div>
    </li>
  );
}
