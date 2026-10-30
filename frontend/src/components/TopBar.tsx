import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useApp } from "../state";

export default function TopBar() {
  const { user, signOut, plan } = useApp();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const nav = useNavigate();

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", close); };
  }, [open]);

  if (!user) return null;
  const links =
    user.role === "farmer"
      ? [["/sell", "Plan a sale"], ["/bookings", "My bookings"]]
      : user.role === "provider"
        ? [["/provider", "Your services"]]
        : [["/admin", "Administration"]];

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <Link to="/" className="wordmark" aria-label="AgriPrice Advisor home">
          <span className="stencil">AP</span>
          <span className="hide-sm">AgriPrice Advisor</span>
        </Link>
        <nav className="topnav" aria-label="Main">
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} className={to === "/bookings" ? "hide-sm" : undefined}>{label}</NavLink>
          ))}
          <div className="menu" ref={ref}>
            <button className="link" aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(!open)}>
              Account
            </button>
            {open && (
              <div className="menu-panel" role="menu">
                <div className="who">{user.name || user.contact}</div>
                {user.name && <div className="faint">{user.contact}</div>}
                {user.role === "farmer" && (
                  <p className="faint" style={{ marginTop: 8 }}>
                    {plan.market ? `Your market: ${plan.market}` : "No market chosen yet"}
                  </p>
                )}
                <div className="stack" style={{ marginTop: 12 }}>
                  {user.role === "farmer" && (
                    <Link className="btn small block" to="/bookings" onClick={() => setOpen(false)}>My bookings</Link>
                  )}
                  <button className="btn small block danger" onClick={() => { signOut(); nav("/"); }}>Sign out</button>
                </div>
              </div>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
