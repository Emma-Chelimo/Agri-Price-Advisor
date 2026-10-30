import { useEffect, useState, type FormEvent } from "react";
import { Link, Navigate } from "react-router-dom";
import { api, monthYear, num, pct, perKg, ugx, type Advice } from "../api";
import { adviceRequest, useApp } from "../state";
import Verdict from "../components/Verdict";
import PriceChart from "../components/PriceChart";
import Prose from "../components/Prose";

export default function AdvicePage() {
  const { plan, meta } = useApp();
  const req = adviceRequest(plan);
  const [a, setA] = useState<Advice | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify(req);

  useEffect(() => {
    if (!req) return;
    setA(null); setError(null);
    api<Advice>("/api/advice", { body: req }).then(setA).catch((e) => setError(e.message));
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!req) return <Navigate to="/sell" replace />;
  if (error) return (
    <div className="page narrow">
      <p className="notice warn">{error}</p>
      <Link className="btn" to="/sell">Change my details</Link>
    </div>
  );
  if (!a) return <div className="page"><p className="faint" aria-live="polite">Working out your advice…</p></div>;

  const latestNote = a.latest.months_old && a.latest.months_old > 1
    ? `The latest recorded ${a.crop.toLowerCase()} price in ${a.market} is from ${monthYear(a.latest.date)}, ${a.latest.months_old} months before the newest data.`
    : null;

  return (
    <div className="page">
      <p style={{ marginBottom: 16 }}>
        <span className="muted">{a.quantity_kg.toLocaleString()} kg of {a.crop.toLowerCase()} at {a.market}. </span>
        <Link to="/sell">Change</Link>
      </p>

      <div className="split">
        <div className="sticky stack">
          <Verdict a={a} />
          {a.latest.is_stale && <p className="notice warn">Prices for this market haven't been recorded for a while, so treat this advice with extra care.</p>}
          <div className="row actions">
            <Link className="btn primary" to="/services">Find {a.advice.months > 0 ? "storage" : a.comparison.go_to_alternative ? "transport or a buyer" : "a buyer"}</Link>
            <Link className="btn" to="/bookings">My bookings</Link>
          </div>
        </div>

        <div className="stack">
          <section>
            <h2>Price in {a.market}</h2>
            <p className="muted">
              Last recorded <strong className="num">{perKg(a.latest.price)}</strong> in {monthYear(a.latest.date)}.
              {" "}{a.forecast.label.replace(/\s*\([^)]*\)\s*$/, "")}.
            </p>
            {latestNote && <p className="faint">{latestNote}</p>}
            <PriceChart a={a} />
            {a.forecast.note && <p className="faint" style={{ marginTop: 8 }}>{a.forecast.note}</p>}
          </section>

          <Markets a={a} />
          <StorageOptions a={a} />
          <Explanation a={a} assistantOn={!!meta?.assistant.enabled} />
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ markets
function Markets({ a }: { a: Advice }) {
  const c = a.comparison;
  const home = c.markets.find((m) => m.is_home);
  const alt = c.best_alternative;
  return (
    <section className="section">
      <h2>Where to sell</h2>
      {!alt || !home ? (
        <p className="muted">No other market has a {a.crop.toLowerCase()} price for {monthYear(c.reference_month + (c.reference_month.length === 7 ? "-01" : ""))} to compare with.</p>
      ) : (
        <>
          <p className="muted">
            {c.go_to_alternative
              ? `${alt.market} pays more, even after the cost of getting there.`
              : `${a.market} is still your best choice once transport is paid for.`}
          </p>
          <div className="compare">
            {[home, alt].map((m) => {
              const better = c.go_to_alternative ? m === alt : m === home;
              return (
                <div key={m.market} className={better ? "better" : undefined}>
                  <strong style={{ fontSize: "var(--fs-md)" }}>{m.market}</strong>
                  <div className="faint">{m.is_home ? "Your market" : `${num(m.road_km)} km by road`}</div>
                  <dl>
                    <dt>Price</dt><dd>{perKg(m.latest_price)}</dd>
                    <dt>Transport</dt><dd>{m.is_home ? "none" : ugx(m.transport_ugx)}</dd>
                    <dt>You get</dt><dd>{ugx(m.sell_now_net_ugx)}</dd>
                  </dl>
                </div>
              );
            })}
          </div>
          <details style={{ marginTop: 12 }}>
            <summary>All {c.markets.length} markets with a price this month</summary>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Market</th><th>Road km</th><th>UGX/kg</th><th>You get</th><th>vs. {a.market}</th></tr></thead>
                <tbody>
                  {c.markets.map((m) => (
                    <tr key={m.market} className={m.market === alt.market ? "best" : undefined}>
                      <td>{m.market}{m.sudden_jump && <span className="faint"> (sudden jump, check first)</span>}</td>
                      <td>{m.is_home ? "0" : num(m.road_km)}</td>
                      <td>{num(m.latest_price)}</td>
                      <td>{num(m.sell_now_net_ugx)}</td>
                      <td className={m.vs_home_sell_now_ugx > 0 ? "up" : m.vs_home_sell_now_ugx < 0 ? "down" : undefined}>
                        {m.is_home ? "" : `${m.vs_home_sell_now_ugx > 0 ? "+" : ""}${num(m.vs_home_sell_now_ugx)}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {c.excluded.length > 0 && <p className="faint">Left out because their latest price is older: {c.excluded.join(", ")}.</p>}
          </details>
        </>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ storage
function StorageOptions({ a }: { a: Advice }) {
  const opts = a.advice.options;
  const now = opts[0];
  return (
    <section className="section">
      <h2>If you store in {a.market}</h2>
      <p className="muted">
        Storing pays only when the price climbs above the break-even, which covers storage at {num(a.costs.storage_ugx_per_kg_month)} UGX/kg a month
        and losing {a.costs.loss_pct_per_month.toFixed(1)}% of the crop each month.
      </p>
      <ul className="list">
        {opts.slice(1).map((o) => {
          const chosen = o.months === a.advice.months;
          return (
            <li key={o.months} className="list-row" style={chosen ? { background: "#fdf6dc" } : undefined}>
              <div>
                <strong>{o.action}</strong>{chosen && <span className="pill" style={{ marginLeft: 8 }}>Advised</span>}
                <div className="faint">Needs at least {perKg(o.breakeven_price)}. Chance of reaching it: {pct(o.prob_beats_now)}.</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="figure">{ugx(o.net_ugx)}</div>
                <div className={`faint ${o.gain_vs_now_ugx >= 0 ? "up" : "down"}`}>
                  {o.gain_vs_now_ugx >= 0 ? "+" : ""}{num(o.gain_vs_now_ugx)} vs. selling now
                </div>
              </div>
            </li>
          );
        })}
      </ul>
      <p className="faint" style={{ marginTop: 8 }}>Selling now at {a.market} gives {ugx(now.net_ugx)}.</p>
    </section>
  );
}

// ------------------------------------------------------------------ explanation and questions
interface Turn { role: "user" | "assistant"; content: string }

function Explanation({ a, assistantOn }: { a: Advice; assistantOn: boolean }) {
  const { plan } = useApp();
  const req = adviceRequest(plan)!;
  const [text, setText] = useState(a.explanation_template);
  const [source, setSource] = useState<"template" | "ai" | "loading">(assistantOn ? "loading" : "template");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    if (!assistantOn) return;
    let live = true;
    api<{ ok: boolean; text: string; source: "ai" | "template" }>("/api/assistant/explain", { body: req })
      .then((r) => { if (live) { setText(r.text); setSource(r.source); } })
      .catch(() => live && setSource("template"));
    return () => { live = false; };
  }, [a]); // eslint-disable-line react-hooks/exhaustive-deps

  async function ask(e: FormEvent) {
    e.preventDefault();
    const question = q.trim();
    if (!question) return;
    setAsking(true); setAskError(null);
    const history = turns;
    setTurns([...history, { role: "user", content: question }]); setQ("");
    try {
      const r = await api<{ ok: boolean; text: string }>("/api/assistant/ask", { body: { advice: req, question, history } });
      setTurns((t) => [...t, { role: "assistant", content: r.text }]);
    } catch (err) {
      setAskError((err as Error).message);
      setTurns(history); setQ(question);
    } finally { setAsking(false); }
  }

  return (
    <section className="section">
      <h2>Why this advice</h2>
      {source === "loading" && <p className="faint" aria-live="polite">Writing a plain-language explanation…</p>}
      <Prose text={text} />
      {source === "ai" && <p className="faint">Written by an AI assistant from the figures above. Numbers it mentions are checked against them.</p>}

      {assistantOn && (
        <>
          <h3 style={{ marginTop: 24 }}>Ask about this advice</h3>
          {turns.length > 0 && (
            <div className="chat" aria-live="polite">
              {turns.map((t, i) => <div key={i} className={`bubble ${t.role === "user" ? "me" : "ai"}`}>{t.role === "user" ? t.content : <Prose text={t.content} />}</div>)}
              {asking && <div className="bubble ai faint">Thinking…</div>}
            </div>
          )}
          <form onSubmit={ask} className="row" style={{ marginTop: 12 }}>
            <input type="text" aria-label="Your question" placeholder="For example: what if I wait until July?" maxLength={500}
              style={{ flex: "1 1 240px", width: "auto" }} value={q} onChange={(e) => setQ(e.target.value)} />
            <button className="btn" disabled={asking || !q.trim()}>Ask</button>
          </form>
          {askError && <p className="error" role="alert">{askError}</p>}
        </>
      )}
    </section>
  );
}
