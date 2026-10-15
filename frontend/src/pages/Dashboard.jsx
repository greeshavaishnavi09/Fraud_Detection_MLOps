
import "../index.css";

const modelMetrics = [
  { label: "Precision", value: "80.45%" },
  { label: "Recall", value: "70.13%" },
  { label: "F1 Score", value: "74.94%" },
  { label: "PR-AUC", value: "82.00%" },
];

const confusionMetrics = [
  { label: "Correctly identified legitimate", value: "952,951" },
  { label: "Legitimate transactions flagged", value: "210" },
  { label: "Fraud transactions missed", value: "368" },
  { label: "Fraud transactions detected", value: "864" },
];

export default function Dashboard() {
  return (
    <main className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <p className="dashboard-eyebrow">FRAUD DETECTION MLOPS</p>
          <h1>Dashboard</h1>
          <p>
            Overview of the fraud detection model and its held-out
            test performance.
          </p>
        </div>
        <span className="dashboard-status">
          Model evaluation complete
        </span>
      </header>

      <section className="dashboard-section">
        <h2>Model Performance</h2>
        <p className="dashboard-description">
          XGBoost results on the held-out test dataset at threshold 0.97.
        </p>

        <div className="dashboard-metrics">
          {modelMetrics.map((metric) => (
            <article className="dashboard-metric" key={metric.label}>
              <p>{metric.label}</p>
              <h3>{metric.value}</h3>
            </article>
          ))}
        </div>
      </section>

      <section className="dashboard-section">
        <h2>Test Evaluation Summary</h2>

        <div className="dashboard-list">
          {confusionMetrics.map((metric) => (
            <div className="dashboard-list-row" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="dashboard-note">
        <h2>About these metrics</h2>
        <p>
          These values come from the saved offline test evaluation.
          They are not live production transaction counts. The
          dashboard can be connected to backend analytics endpoints
          as those features are implemented.
        </p>
      </section>
    </main>
  );
}
