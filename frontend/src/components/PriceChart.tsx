import { useMemo, useState } from "react";
import type { Advice } from "../api";
import { monthYear, num } from "../api";

// Past prices (solid green), the forecast (dashed ink) and its likely range (gold band).
// Hand-drawn SVG so it stays light on a phone and needs no chart library.

const W = 720, H = 300, PAD = { l: 56, r: 16, t: 16, b: 32 };
const MONTHS_SHOWN = 36;

export default function PriceChart({ a }: { a: Advice }) {
  const [hover, setHover] = useState<number | null>(null);

  const d = useMemo(() => {
    const hist = a.history.slice(-MONTHS_SHOWN).map((h) => ({ t: h.date, p: h.price, observed: h.observed }));
    const fc = a.forecast.points;
    const last = hist[hist.length - 1];
    const all = [...hist.map((h) => h.p), ...fc.map((f) => f.high ?? f.price), ...fc.map((f) => f.low ?? f.price)];
    let lo = Math.min(...all), hi = Math.max(...all);
    const padY = (hi - lo) * 0.08 || hi * 0.1;
    lo = Math.max(0, lo - padY); hi = hi + padY;
    const n = hist.length + fc.length;
    const x = (i: number) => PAD.l + (i / Math.max(1, n - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
    const histPath = hist.map((h, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(h.p).toFixed(1)}`).join("");
    const i0 = hist.length - 1;
    const fcPath = `M${x(i0)},${y(last.p)}` + fc.map((f, k) => `L${x(i0 + k + 1).toFixed(1)},${y(f.price).toFixed(1)}`).join("");
    const band = fc.some((f) => f.low != null)
      ? `M${x(i0)},${y(last.p)}` +
        fc.map((f, k) => `L${x(i0 + k + 1).toFixed(1)},${y(f.high ?? f.price).toFixed(1)}`).join("") +
        [...fc].reverse().map((f, k) => `L${x(i0 + fc.length - k).toFixed(1)},${y(f.low ?? f.price).toFixed(1)}`).join("") + "Z"
      : null;
    // y ticks: 4 round numbers
    const step = niceStep((hi - lo) / 4);
    const ticks: number[] = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
    // x ticks: each January
    const points = [...hist.map((h) => ({ t: h.t, p: h.p, low: null as number | null, high: null as number | null, fc: false, observed: h.observed })),
      ...fc.map((f) => ({ t: f.date, p: f.price, low: f.low, high: f.high, fc: true, observed: false }))];
    const years = points.map((p, i) => ({ i, t: p.t })).filter((p) => p.t.slice(5, 7) === "01");
    return { hist, fc, x, y, histPath, fcPath, band, ticks, years, points, i0 };
  }, [a]);

  const hp = hover != null ? d.points[hover] : null;

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const n = d.points.length;
    const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1));
    setHover(Math.min(n - 1, Math.max(0, i)));
  }

  return (
    <figure className="chart" style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        aria-label={`${a.crop} price in ${a.market}: ${num(d.hist[d.hist.length - 1].p)} UGX per kg in ${monthYear(d.hist[d.hist.length - 1].t)}, forecast ${num(d.fc[d.fc.length - 1]?.price)} UGX per kg by ${d.fc.length ? monthYear(d.fc[d.fc.length - 1].date) : ""}.`}>
        <g className="axis">
          {d.ticks.map((v) => (
            <g key={v}>
              <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={d.y(v)} y2={d.y(v)} />
              <text x={PAD.l - 8} y={d.y(v) + 4} textAnchor="end">{num(v)}</text>
            </g>
          ))}
          {d.years.map((p) => (
            <text key={p.t} x={d.x(p.i)} y={H - 10} textAnchor="middle">{p.t.slice(0, 4)}</text>
          ))}
        </g>
        <rect x={d.x(d.i0)} y={PAD.t} width={W - PAD.r - d.x(d.i0)} height={H - PAD.t - PAD.b} fill="#17211a" opacity="0.035" />
        {d.band && <path d={d.band} fill="var(--band)" />}
        <path d={d.histPath} fill="none" stroke="var(--green)" strokeWidth="2.5" strokeLinejoin="round" />
        <path d={d.fcPath} fill="none" stroke="var(--ink)" strokeWidth="2.5" strokeDasharray="6 5" />
        {d.hist.map((h, i) => !h.observed ? null : (
          <circle key={h.t} cx={d.x(i)} cy={d.y(h.p)} r="2.2" fill="var(--green)" />
        ))}
        <circle cx={d.x(d.i0)} cy={d.y(d.hist[d.i0].p)} r="5" fill="var(--green)" stroke="#fff" strokeWidth="2" />
        {hp && hover != null && (
          <g pointerEvents="none">
            <line x1={d.x(hover)} x2={d.x(hover)} y1={PAD.t} y2={H - PAD.b} stroke="var(--ink-faint)" />
            <circle cx={d.x(hover)} cy={d.y(hp.p)} r="5" fill={hp.fc ? "var(--ink)" : "var(--green)"} stroke="#fff" strokeWidth="2" />
          </g>
        )}
      </svg>
      <div className="row" style={{ justifyContent: "space-between", minHeight: 28 }}>
        <div className="legend">
          <span><i className="key" style={{ background: "var(--green)", height: 3 }} />Recorded price</span>
          <span><i className="key" style={{ borderTop: "3px dashed var(--ink)", height: 0 }} />Forecast</span>
          {d.band && <span><i className="key" style={{ background: "var(--band)" }} />Likely range</span>}
        </div>
        <div className="faint num" aria-live="polite">
          {hp ? `${monthYear(hp.t)}: ${num(hp.p)} UGX/kg${hp.fc && hp.low != null ? ` (likely ${num(hp.low)} to ${num(hp.high)})` : ""}${!hp.fc && !hp.observed ? ", filled gap" : ""}` : "Touch the chart to read a month"}
        </div>
      </div>
    </figure>
  );
}

function niceStep(raw: number) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
