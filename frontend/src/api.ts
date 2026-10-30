// Small client for the AgriPrice Advisor API (api/main.py).

const TOKEN_KEY = "agriprice.token";

export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(t: string | null) {
  try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function api<T>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let res: Response;
  try {
    res = await fetch(path, {
      method: opts.method ?? (opts.body !== undefined ? "POST" : "GET"),
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that it is running and try again.");
  }
  if (!res.ok) {
    let msg = `Something went wrong (error ${res.status}).`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = "Please check the details you entered.";
    } catch { /* not JSON */ }
    if (res.status === 401) setToken(null);
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------------ types
export type Role = "farmer" | "provider" | "admin";
export interface User { role: Role; contact: string; name: string; }

export interface Meta {
  crops: string[];
  reference_month: string;
  max_horizon: number;
  service_kinds: Record<string, { label: string; price_unit: string }>;
  quote_deadline_hours: number;
  assistant: { enabled: boolean; model: string | null };
  all_markets: string[];
}

export interface MarketChoice { market: string | null; source: "manual" | "location" | "none"; distance_km: number | null; note: string; }

export interface AdviceRequest {
  crop: string; market: string; quantity_kg: number; horizon?: number;
  storage?: number | null; loss_pct?: number | null; transport_rate?: number | null;
}

export interface Option {
  action: string; months: number; price: number; low: number; high: number; kg_sold: number;
  storage_ugx: number; transport_ugx: number; net_ugx: number; gain_vs_now_ugx: number;
  breakeven_price: number | null; prob_beats_now: number | null;
}

export interface MarketRow {
  market: string; region: string; is_home: boolean; road_km: number; latest_price: number;
  transport_ugx: number; best_option: string; best_months: number; best_net_ugx: number;
  sell_now_net_ugx: number; vs_home_sell_now_ugx: number; sudden_jump: boolean;
}

export interface Advice {
  crop: string; market: string; quantity_kg: number; horizon: number;
  latest: { price: number; date: string; months_old: number | null; is_stale: boolean };
  history: { date: string; price: number; observed: boolean }[];
  forecast: {
    method: string; label: string; level: number; note: string; used_fallback: boolean;
    points: { date: string; price: number; low: number | null; high: number | null; months_ahead: number; typical_error_pct: number | null }[];
  };
  costs: { storage_ugx_per_kg_month: number; loss_pct_per_month: number; transport_ugx_per_kg_km: number };
  advice: {
    recommendation: string; months: number; expected_net_ugx: number; net_sell_now_ugx: number; gain_ugx: number;
    confidence: number | null; confidence_label: string; best_store_months: number | null; best_store_prob: number | null;
    store_min_probability: number; options: Option[];
  };
  comparison: { reference_month: string; excluded: string[]; markets: MarketRow[]; best_alternative: MarketRow | null; go_to_alternative: boolean };
  sell_market: string;
  explanation_template: string;
}

export interface Provider {
  id: number; name: string; kind: string; market: string; crops: string; capacity_kg: number; price: number;
  description: string; distance_km: number; estimate_ugx: number; rating: number | null; rating_count: number; trips?: number;
}

export interface BookingEvent { at: string; actor: string; status: string | null; label: string; note: string; condition: string; }
export interface Booking {
  id: number; kind: string; crop: string; quantity_kg: number; start_date: string; end_date: string | null;
  origin_market: string; destination_market: string | null; share_transport: boolean; status: string; status_label: string;
  quote_ugx: number | null; quote_note: string; requested_at: string; provider_id: number; provider_name: string;
  provider_market: string; rating_stars: number | null; price_received_ugx_per_kg: number | null;
  contacts_visible: boolean; provider_contact?: string | null; farmer_contact?: string | null; map_link?: string | null;
  reply_by?: string; events: BookingEvent[];
  next_option?: { id: number; name: string; market: string; kind: string; price: number } | null;
  recheck?: { recommendation: string; price: number; date: string; confidence_label: string };
}

// ------------------------------------------------------------------ formatting
const n0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const ugx = (v: number | null | undefined) => (v == null ? "n/a" : `${n0.format(v)} UGX`);
export const perKg = (v: number | null | undefined) => (v == null ? "n/a" : `${n0.format(v)} UGX/kg`);
export const num = (v: number | null | undefined) => (v == null ? "n/a" : n0.format(v));
export const pct = (p: number | null | undefined) => (p == null ? "n/a" : `${Math.round(p * 100)}%`);
export const monthYear = (iso: string) =>
  new Date(iso + (iso.length === 10 ? "T00:00:00" : "")).toLocaleDateString("en-GB", { month: "long", year: "numeric" });
export const dayMonth = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
export const dateTime = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
