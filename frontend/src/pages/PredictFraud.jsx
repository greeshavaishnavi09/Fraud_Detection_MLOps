
import { useState } from "react";
import { predictFraud } from "../api/predictionApi";

const initialForm = {
  step: "100",
  amount: "5000",
  oldbalanceOrg: "10000",
  type: "TRANSFER",
};

export default function PredictFraud() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const transaction = {
        step: Number(form.step),
        amount: Number(form.amount),
        oldbalanceOrg: Number(form.oldbalanceOrg),
        type: form.type,
      };

      if (
        !Number.isInteger(transaction.step) ||
        transaction.step < 0
      ) {
        throw new Error("Step must be a non-negative integer.");
      }

      if (
        !Number.isFinite(transaction.amount) ||
        transaction.amount < 0
      ) {
        throw new Error("Amount must be a non-negative number.");
      }

      if (
        !Number.isFinite(transaction.oldbalanceOrg) ||
        transaction.oldbalanceOrg < 0
      ) {
        throw new Error(
          "Origin balance must be a non-negative number."
        );
      }

      const data = await predictFraud(transaction);
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to get prediction.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="prediction-page">
      <header className="prediction-header">
        <p className="prediction-eyebrow">FRAUD DETECTION MLOPS</p>
        <h1>Transaction Risk Prediction</h1>
        <p>
          Enter transaction details to estimate the probability
          of fraudulent activity.
        </p>
      </header>

      <section className="prediction-card">
        <h2>Transaction Details</h2>

        <form onSubmit={handleSubmit}>
          <div className="prediction-grid">
            <label>
              Transaction Step
              <input
                type="number"
                name="step"
                min="0"
                step="1"
                value={form.step}
                onChange={handleChange}
                required
              />
            </label>

            <label>
              Transaction Type
              <select
                name="type"
                value={form.type}
                onChange={handleChange}
                required
              >
                <option value="CASH_IN">CASH_IN</option>
                <option value="CASH_OUT">CASH_OUT</option>
                <option value="DEBIT">DEBIT</option>
                <option value="PAYMENT">PAYMENT</option>
                <option value="TRANSFER">TRANSFER</option>
              </select>
            </label>

            <label>
              Transaction Amount
              <input
                type="number"
                name="amount"
                min="0"
                step="any"
                value={form.amount}
                onChange={handleChange}
                required
              />
            </label>

            <label>
              Original Account Balance
              <input
                type="number"
                name="oldbalanceOrg"
                min="0"
                step="any"
                value={form.oldbalanceOrg}
                onChange={handleChange}
                required
              />
            </label>
          </div>

          <button
            type="submit"
            className="prediction-button"
            disabled={loading}
          >
            {loading ? "Analyzing Transaction..." : "Predict Fraud"}
          </button>
        </form>

        {error && (
          <div className="prediction-error" role="alert">
            {error}
          </div>
        )}

        {result && (
          <section className="prediction-result" aria-live="polite">
            <h2>Prediction Result</h2>

            <p>
              Classification:{" "}
              <strong>
                {result.is_fraud ? "Potential Fraud" : "Not Flagged"}
              </strong>
            </p>

            <p>
              Estimated Fraud Probability:{" "}
              <strong>
                {(result.fraud_probability * 100).toFixed(2)}%
              </strong>
            </p>

            <p>
              Decision Threshold:{" "}
              <strong>{(result.threshold * 100).toFixed(0)}%</strong>
            </p>

            <p>
              Risk Classification:{" "}
              <strong>{result.risk_level}</strong>
            </p>

            <p className="prediction-disclaimer">
              This is a model estimate, not a guarantee that a
              transaction is safe or fraudulent.
            </p>
          </section>
        )}
      </section>
    </main>
  );
}
