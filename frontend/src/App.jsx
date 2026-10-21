
import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import PredictFraud from "./pages/PredictFraud";

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");

  return (
    <div className="app-layout">
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">F</div>
          <div>
            <strong>Sentinel AI</strong>
            <span>FRAUD INTELLIGENCE</span>
          </div>
        </div>

        <p className="sidebar-section-label">WORKSPACE</p>

        <nav className="sidebar-navigation">
          <button
            className={`sidebar-link ${activePage === "dashboard" ? "active" : ""}`}
            onClick={() => setActivePage("dashboard")}
          >
            <span className="sidebar-icon">▦</span>
            Dashboard
          </button>

          <button
            className={`sidebar-link ${activePage === "predict" ? "active" : ""}`}
            onClick={() => setActivePage("predict")}
          >
            <span className="sidebar-icon">◎</span>
            Predict Fraud
          </button>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <div>
            <strong>Prediction API</strong>
            <span>Local development</span>
          </div>
        </div>
      </aside>

      <div className="app-main">
        <header className="app-topbar">
          <div>
            <p className="breadcrumb">Workspace / Fraud Intelligence</p>
            <h1>
              {activePage === "dashboard"
                ? "Dashboard"
                : "Fraud Prediction"}
            </h1>
          </div>
          <div className="topbar-status">
            <span className="status-dot" />
            Development environment
          </div>
        </header>

        <div className="app-content">
          {activePage === "dashboard" ? (
            <Dashboard />
          ) : (
            <PredictFraud />
          )}
        </div>
      </div>
    </div>
  );
}
