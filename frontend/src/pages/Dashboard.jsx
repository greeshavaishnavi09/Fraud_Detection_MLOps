
const metrics = [
  { label: "Precision", value: "80.45%", detail: "Fraud alerts that were correct" },
  { label: "Recall", value: "70.13%", detail: "Fraud cases detected" },
  { label: "F1 Score", value: "74.94%", detail: "Precision and recall balance" },
  { label: "PR-AUC", value: "82.00%", detail: "Precision-recall performance" },
];

const summary = [
  { label: "Legitimate transactions correctly identified", value: "952,951" },
  { label: "Legitimate transactions flagged", value: "210" },
  { label: "Fraud transactions missed", value: "368" },
  { label: "Fraud transactions detected", value: "864" },
];

export default function Dashboard() {
  return (
    <div className="dashboard-page">
      <section className="dashboard-intro">
        <p className="eyebrow">FRAUD DETECTION MLOPS</p>
        <h2>Model performance overview</h2>
        <p>
          Monitor the evaluated performance of your XGBoost fraud
          detection model.
        </p>
        <span className="evaluation-status">
          <span className="status-dot" />
          Offline test evaluation complete
        </span>
      </section>

      <section className="dashboard-metrics">
        {metrics.map((metric) => (
          <article className="dashboard-metric-card" key={metric.label}>
            <p>{metric.label}</p>
            <h3>{metric.value}</h3>
            <span>{metric.detail}</span>
          </article>
        ))}
      </section>

      <section className="dashboard-summary-card">
        <div className="section-heading">
          <div>
            <h3>Test evaluation summary</h3>
            <p>Held-out test set · Decision threshold 0.97</p>
          </div>
          <span className="model-tag">XGBoost</span>
        </div>

        <div className="summary-grid">
          {summary.map((item, index) => (
            <div className="summary-item" key={item.label}>
              <span className="summary-number">
                {item.value}
              </span>
              <span className="summary-label">{item.label}</span>
            </div>
          ))}
        </div>
      </section>

      <p className="dashboard-disclaimer">
        These figures are from the saved offline test evaluation, not
        live production transaction counts. Live monitoring will be
        connected separately.
      </p>
    </div>
  );
}
