import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Role, type User } from "../api";
import { useApp } from "../state";

const ROLES: { value: Role; label: string; hint: string }[] = [
  { value: "farmer", label: "Farmer", hint: "Get selling advice" },
  { value: "provider", label: "Service provider or buyer", hint: "Storage, transport, buying" },
  { value: "admin", label: "Administrator", hint: "Approve providers" },
];

export default function SignIn() {
  const { signIn, meta } = useApp();
  const nav = useNavigate();
  const [role, setRole] = useState<Role>("farmer");
  const [contact, setContact] = useState("");
  const [name, setName] = useState("");
  const [consent, setConsent] = useState(false);
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const body = role === "admin" ? { role, password } : { role, contact, name, consent };
      const r = await api<{ token: string; user: User }>("/api/auth/signin", { body });
      signIn(r.token, r.user);
      nav(role === "farmer" ? "/sell" : role === "provider" ? "/provider" : "/admin", { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally { setBusy(false); }
  }

  return (
    <div className="signin">
      <div className="signin-brand">
        <div className="wordmark" style={{ color: "var(--ink)" }}>AgriPrice Advisor</div>
        <h1>Sell now or store?</h1>
        <p style={{ fontSize: "var(--fs-md)", maxWidth: "34ch", margin: 0 }}>
          Price forecasts for maize, beans, sorghum, millet and flour in Ugandan markets, with the costs of storing
          and moving your harvest worked in.
          {meta && <> Prices recorded up to {new Date(meta.reference_month + "-01").toLocaleDateString("en-GB", { month: "long", year: "numeric" })}.</>}
        </p>
      </div>

      <div className="signin-form">
        <div>
          <h2>Sign in</h2>
          <form onSubmit={submit} className="stack" noValidate>
            <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
              <legend style={{ fontWeight: 700, marginBottom: 8 }}>I am a</legend>
              <div className="choices" style={{ gridTemplateColumns: "1fr" }}>
                {ROLES.map((r) => (
                  <label key={r.value} className="choice">
                    <input type="radio" name="role" value={r.value} checked={role === r.value} onChange={() => { setRole(r.value); setError(null); }} />
                    <span>{r.label}<small>{r.hint}</small></span>
                  </label>
                ))}
              </div>
            </fieldset>

            {role === "admin" ? (
              <label className="field">
                <span>Administrator password</span>
                <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
              </label>
            ) : (
              <>
                <label className="field">
                  <span>Phone number or email</span>
                  <input type="text" inputMode="email" autoComplete="username" placeholder="0772 123 456"
                    value={contact} onChange={(e) => setContact(e.target.value)} />
                  <small>A Ugandan mobile number or an email address. Providers use it to reach you once you accept a quote.</small>
                </label>
                <label className="field">
                  <span>Your name <span className="faint">(optional)</span></span>
                  <input type="text" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} />
                </label>
                <label className="check">
                  <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                  <span>I agree that AgriPrice Advisor keeps my contact to manage my requests, and shares it with a provider only after I accept their quote.</span>
                </label>
              </>
            )}

            {error && <p className="error" role="alert">{error}</p>}
            <button className="btn primary block" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
          </form>
        </div>
      </div>
    </div>
  );
}
