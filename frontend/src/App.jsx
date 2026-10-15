
import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import PredictFraud from "./pages/PredictFraud";

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");

  return (
    <div className="app-shell">
      <nav className="app-navbar">
        <div className="app-brand">Fraud Detection MLOps</div>

        <div className="app-nav-links">
          <button
            type="button"
            className={activePage === "dashboard" ? "active" : ""}
            onClick={() => setActivePage("dashboard")}
          >
            Dashboard
          </button>

          <button
            type="button"
            className={activePage === "predict" ? "active" : ""}
            onClick={() => setActivePage("predict")}
          >
            Predict Fraud
          </button>
        </div>
      </nav>

      {activePage === "dashboard" ? <Dashboard /> : <PredictFraud />}
    </div>
  );
}
