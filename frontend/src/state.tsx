// Shared app state: who is signed in, reference data, and the farmer's current plan.
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, getToken, setToken, type AdviceRequest, type Meta, type User } from "./api";

const PLAN_KEY = "agriprice.plan";

export interface Plan {
  crop: string | null;
  quantity_kg: number;
  market: string | null;
  marketNote: string;
  storage: number | null;
  loss_pct: number | null;
  transport_rate: number | null;
}
const emptyPlan: Plan = { crop: null, quantity_kg: 500, market: null, marketNote: "", storage: null, loss_pct: null, transport_rate: null };

function loadPlan(): Plan {
  try {
    const raw = localStorage.getItem(PLAN_KEY);
    return raw ? { ...emptyPlan, ...JSON.parse(raw) } : emptyPlan;
  } catch { return emptyPlan; }
}

export function adviceRequest(p: Plan): AdviceRequest | null {
  if (!p.crop || !p.market || !(p.quantity_kg > 0)) return null;
  return { crop: p.crop, market: p.market, quantity_kg: p.quantity_kg,
    storage: p.storage, loss_pct: p.loss_pct, transport_rate: p.transport_rate };
}

interface AppState {
  user: User | null;
  ready: boolean;
  meta: Meta | null;
  metaError: string | null;
  plan: Plan;
  setPlan: (p: Partial<Plan>) => void;
  signIn: (token: string, user: User) => void;
  signOut: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [plan, setPlanState] = useState<Plan>(loadPlan);

  useEffect(() => {
    api<Meta>("/api/meta").then(setMeta).catch((e) => setMetaError(e.message));
    if (!getToken()) { setReady(true); return; }
    api<User>("/api/auth/me").then(setUser).catch(() => setToken(null)).finally(() => setReady(true));
  }, []);

  const setPlan = useCallback((p: Partial<Plan>) => {
    setPlanState((old) => {
      const next = { ...old, ...p };
      try { localStorage.setItem(PLAN_KEY, JSON.stringify(next)); } catch { /* ignore */ }
      return next;
    });
  }, []);

  const signIn = useCallback((token: string, u: User) => { setToken(token); setUser(u); }, []);
  const signOut = useCallback(() => { setToken(null); setUser(null); }, []);

  return (
    <Ctx.Provider value={{ user, ready, meta, metaError, plan, setPlan, signIn, signOut }}>
      {children}
    </Ctx.Provider>
  );
}

export function useApp(): AppState {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}
