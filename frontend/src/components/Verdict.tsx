import type { Advice } from "../api";
import { pct, ugx } from "../api";

// The one bold element: the decision stencilled on maize gold, plus how sure we are.
export default function Verdict({ a }: { a: Advice }) {
  const adv = a.advice;
  const store = adv.months > 0;
  const alt = a.comparison.go_to_alternative ? a.comparison.best_alternative : null;
  const amount = store ? adv.expected_net_ugx : alt ? alt.sell_now_net_ugx : adv.net_sell_now_ugx;
  const word = store ? `Store ${adv.months} month${adv.months > 1 ? "s" : ""}` : "Sell now";
  const where = a.sell_market !== a.market ? ` in ${a.sell_market}` : ` in ${a.market}`;
  const line = store
    ? `Keep your ${a.quantity_kg.toLocaleString()} kg of ${a.crop.toLowerCase()} and sell${where} later. Expected after costs:`
    : `Sell your ${a.quantity_kg.toLocaleString()} kg of ${a.crop.toLowerCase()}${where} this month. Expected after costs:`;

  return (
    <section className="verdict" aria-labelledby="verdict-word">
      <h1 id="verdict-word" className="verdict-word">{word}</h1>
      <p className="verdict-line">{line}</p>
      <div className="verdict-amount num">{ugx(amount)}</div>
      <ConfidenceMeter p={store ? adv.confidence : adv.best_store_prob} label={adv.confidence_label.split(":")[0]}
        threshold={adv.store_min_probability} store={store} bestMonths={adv.best_store_months} />
    </section>
  );
}

function ConfidenceMeter({ p, label, threshold, store, bestMonths }:
  { p: number | null; label: string; threshold: number; store: boolean; bestMonths: number | null }) {
  // For "store", p is the chance storing beats selling now.
  // For "sell now", p is the chance the best storage option would have won; low is good.
  const value = p ?? 0;
  const text = store
    ? `${pct(p)} chance that storing pays more than selling now.`
    : bestMonths
      ? `Storing ${bestMonths} month${bestMonths > 1 ? "s" : ""} has only a ${pct(p)} chance of paying more. We advise storing only from ${pct(threshold)}.`
      : "No storage option is expected to pay more than selling now.";
  return (
    <div className="verdict-foot">
      <div className="meter" role="img" aria-label={text}>
        <div className="meter-track">
          <div className="meter-fill" style={{ width: `${Math.round(value * 100)}%` }} />
          <div className="meter-mark" style={{ left: `${threshold * 100}%` }} title={`Store from ${pct(threshold)}`} />
        </div>
        <div className="meter-legend"><span>0%</span><span className="meter-label" style={{ left: `${threshold * 100}%` }}>Store from {pct(threshold)}</span><span>100%</span></div>
      </div>
      <p style={{ margin: "8px 0 0" }}><strong>How sure: {label.toLowerCase()}.</strong> {text}</p>
    </div>
  );
}
