import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import { AppProvider, useApp } from "./state";
import type { Role } from "./api";
import TopBar from "./components/TopBar";
import SignIn from "./pages/SignIn";
import Sell from "./pages/Sell";
import AdvicePage from "./pages/Advice";
import Provider from "./pages/Provider";
import Admin from "./pages/Admin";
import Services from "./pages/Services";
import Bookings from "./pages/Bookings";

function home(role: Role) {
  return role === "farmer" ? "/sell" : role === "provider" ? "/provider" : "/admin";
}

function Guard({ role, children }: { role: Role; children: ReactNode }) {
  const { user, ready } = useApp();
  const loc = useLocation();
  if (!ready) return <div className="page faint">Loading…</div>;
  if (!user) return <Navigate to="/" replace state={{ from: loc.pathname }} />;
  if (user.role !== role) return <Navigate to={home(user.role)} replace />;
  return (
    <>
      <TopBar />
      <main id="main">{children}</main>
    </>
  );
}

function Start() {
  const { user, ready } = useApp();
  if (!ready) return <div className="page faint">Loading…</div>;
  return user ? <Navigate to={home(user.role)} replace /> : <SignIn />;
}

export default function App() {
  return (
    <AppProvider>
      <Routes>
        <Route path="/" element={<Start />} />
        <Route path="/sell" element={<Guard role="farmer"><Sell /></Guard>} />
        <Route path="/advice" element={<Guard role="farmer"><AdvicePage /></Guard>} />
        <Route path="/services" element={<Guard role="farmer"><Services /></Guard>} />
        <Route path="/bookings" element={<Guard role="farmer"><Bookings /></Guard>} />
        <Route path="/provider" element={<Guard role="provider"><Provider /></Guard>} />
        <Route path="/admin" element={<Guard role="admin"><Admin /></Guard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppProvider>
  );
}
