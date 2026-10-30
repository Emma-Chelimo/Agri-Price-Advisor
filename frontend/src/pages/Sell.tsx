import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type MarketChoice } from "../api";
import { useApp } from "../state";

const QUICK_KG = [100, 500, 1000, 5000];

interface MarketItem { market: string; region: string }
interface Costs { storage_ugx_per_kg_month: number; loss_pct_per_month: number; transport_ugx_per_kg_km: number }

export default function Sell() {
  const { meta, metaError, plan, setPlan, user } = useApp();
  const nav = useNavigate();
  const [markets, setMarkets] = useState<MarketItem[]>([]);
  const [defaults, setDefaults] = useState<Costs | null>(null);
  const [locating, setLocating] = useState(false);
  const [locNote, setLocNote] = useState(plan.marketNote);

  // Markets and default costs depend on the crop.
  useEffect(() => {
    const crop = plan.crop;
    if (!crop) return;
    api<MarketItem[]>(`/api/markets?crop=${encodeURIComponent(crop)}`).then((m) => {
      setMarkets(m);
      if (plan.market && !m.some((x) => x.market === plan.market)) {
        setPlan({ market: null, marketNote: "" });
        setLocNote(`${plan.market} has no recent ${crop.toLowerCase()} prices. Choose another market.`);
      }
    }).catch(() => setMarkets([]));
    api<Costs>(`/api/costs/${encodeURIComponent(crop)}`).then(setDefaults).catch(() => setDefaults(null));
  }, [plan.crop]); // eslint-disable-line react-hooks/exhaustive-deps

  function locate() {
    if (!plan.crop) return;
    if (!("geolocation" in navigator)) { setLocNote("This browser can't share your location. Pick your market from the list."); return; }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const c = await api<MarketChoice>("/api/location/resolve", {
            body: { crop: plan.crop, lat: pos.coords.latitude, lon: pos.coords.longitude } });
          setLocNote(c.note);
          if (c.market) setPlan({ market: c.market, marketNote: c.note });
        } catch (e) { setLocNote((e as Error).message); }
        finally { setLocating(false); }
      },
      () => { setLocating(false); setLocNote("Location was not shared. Pick your market from the list instead."); },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 600000 },
    );
  }

  const ready = plan.crop && plan.market && plan.quantity_kg > 0;
  const firstName = user?.name?.split(" ")[0];

  return (
    <div className="page narrow stack">
      <div>
        <h1>{firstName ? `What are you selling, ${firstName}?` : "What are you selling?"}</h1>
        <p className="muted">Three answers and you'll get advice for this month.</p>
      </div>
      {metaError && <p className="notice warn">{metaError}</p>}

      <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
        <legend><h2 style={{ fontSize: "var(--fs-md)" }}>Crop</h2></legend>
        <div className="choices">
          {(meta?.crops ?? []).map((c) => (
            <label key={c} className="choice">
              <input type="radio" name="crop" checked={plan.crop === c} onChange={() => setPlan({ crop: c, storage: null, loss_pct: null })} />
              {c}
            </label>
          ))}
        </div>
      </fieldset>

      <div>
        <label className="field">
          <span style={{ fontSize: "var(--fs-md)" }}>How many kilograms?</span>
          <input type="number" inputMode="numeric" min={1} max={1000000} value={plan.quantity_kg || ""}
            onChange={(e) => setPlan({ quantity_kg: Number(e.target.value) })} />
        </label>
        <div className="chips" style={{ marginTop: 8 }}>
          {QUICK_KG.map((k) => (
            <button key={k} type="button" className="chip" aria-pressed={plan.quantity_kg === k} onClick={() => setPlan({ quantity_kg: k })}>
              {k.toLocaleString()} kg
            </button>
          ))}
        </div>
        <p className="faint" style={{ marginTop: 8 }}>A 100 kg bag is about one sack of maize grain.</p>
      </div>

      <div>
        <h2 style={{ fontSize: "var(--fs-md)" }}>Your nearest market</h2>
        {!plan.crop ? <p className="faint">Choose a crop first. Not every market records every crop.</p> : (
          <>
            <div className="row">
              <button type="button" className="btn" onClick={locate} disabled={locating}>
                {locating ? "Finding your market…" : "Use my location"}
              </button>
              <span className="faint">or</span>
              <select aria-label="Market" style={{ flex: "1 1 200px", width: "auto" }} value={plan.market ?? ""}
                onChange={(e) => { setPlan({ market: e.target.value || null, marketNote: "" }); setLocNote(""); }}>
                <option value="">Pick from the list</option>
                {markets.map((m) => <option key={m.market} value={m.market}>{m.market} ({m.region})</option>)}
              </select>
            </div>
            {locNote && <p className="notice" style={{ marginTop: 12 }}>{locNote}</p>}
          </>
        )}
      </div>

      {plan.crop && defaults && (
        <details>
          <summary>Storage and transport costs</summary>
          <p className="faint">These are typical figures for {plan.crop.toLowerCase()}. Change them if yours are different.</p>
          <div className="stack">
            <CostField label="Storage, UGX per kg each month" value={plan.storage} fallback={defaults.storage_ugx_per_kg_month}
              onChange={(v) => setPlan({ storage: v })} />
            <CostField label="Crop lost in storage, % each month" value={plan.loss_pct} fallback={defaults.loss_pct_per_month} step={0.1}
              onChange={(v) => setPlan({ loss_pct: v })} />
            <CostField label="Transport, UGX per kg for each km" value={plan.transport_rate} fallback={defaults.transport_ugx_per_kg_km} step={0.5}
              onChange={(v) => setPlan({ transport_rate: v })} />
          </div>
        </details>
      )}

      <div className="actionbar">
        <button className="btn primary block" disabled={!ready} onClick={() => nav("/advice")}>Get my advice</button>
      </div>
    </div>
  );
}

function CostField({ label, value, fallback, step = 1, onChange }:
  { label: string; value: number | null; fallback: number; step?: number; onChange: (v: number | null) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min={0} step={step} value={value ?? fallback}
        onChange={(e) => { const v = e.target.value === "" ? null : Number(e.target.value); onChange(v === fallback ? null : v); }} />
      {value != null && <small><button type="button" className="link" onClick={() => onChange(null)}>Use the typical figure ({fallback})</button></small>}
    </label>
  );
}
