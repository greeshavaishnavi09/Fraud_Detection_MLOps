
import { useState } from "react";

import Dashboard from "./pages/Dashboard";
import PredictFraud from "./pages/PredictFraud";
import Transactions from "./pages/Transactions";
import Analytics from "./pages/Analytics";
import Monitoring from "./pages/Monitoring";
import ModelManagement from "./pages/ModelManagement";

const navigation = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "predict", label: "Predict Fraud", icon: "◎" },
  { id: "transactions", label: "Transactions", icon: "⇄" },
  { id: "analytics", label: "Analytics", icon: "▥" },
  { id: "monitoring", label: "Model Monitoring", icon: "⌁" },
  { id: "models", label: "Model Management", icon: "◇" },
];

const pageTitles = {
  dashboard: "Dashboard",
  predict: "Fraud Prediction",
  transactions: "Transactions",
  analytics: "Analytics",
  monitoring: "Model Monitoring",
  models: "Model Management",
};

function PageContent({ activePage }) {
  switch (activePage) {
    case "predict":
      return <PredictFraud />;
    case "transactions":
      return <Transactions />;
    case "analytics":
      return <Analytics />;
    case "monitoring":
      return <Monitoring />;
    case "models":
      return <ModelManagement />;
    default:
      return <Dashboard />;
  }
}

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
          {navigation.map((item) => (
            <button
              key={item.id}
              type="button"
              className={
                activePage === item.id
                  ? "sidebar-link active"
                  : "sidebar-link"
              }
              onClick={() => setActivePage(item.id)}
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
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
            <h1>{pageTitles[activePage]}</h1>
          </div>

          <div className="topbar-status">
            <span className="status-dot" />
            Development environment
          </div>
        </header>

        <div className="app-content">
          <PageContent activePage={activePage} />
        </div>
      </div>
    </div>
  );
}
