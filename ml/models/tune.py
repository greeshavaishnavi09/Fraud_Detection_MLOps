
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
FEATURES = ["step", "amount", "oldbalanceOrg", "type"]

MODELS = {
    "Logistic Regression": MODEL_DIR / "baseline_logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost (original)": MODEL_DIR / "xgboost.joblib",
    "XGBoost (configured)": MODEL_DIR / "xgboost_configured.joblib",
}

THRESHOLDS = np.arange(0.01, 1.00, 0.01)


# --------------------------------------------------
# 2. Threshold evaluation
# --------------------------------------------------

def main():
    print("Loading validation dataset...")

    validation_df = pd.read_csv(VALIDATION_PATH)
    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET].astype(int)

    del validation_df

    results = []

    for model_name, model_path in MODELS.items():

        if not model_path.exists():
            print(f"\nSkipping {model_name}: model not found.")
            print(f"Expected: {model_path}")
            continue

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)
        probabilities = model.predict_proba(X_val)[:, 1]

        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)

            results.append({
                "model": model_name,
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "f2_score": fbeta_score(
                    y_val, predictions, beta=2, zero_division=0
                ),
                "flagged_transactions": int(predictions.sum()),
            })

        del model, probabilities

    if not results:
        raise FileNotFoundError(
            "No candidate models were found. Check models/candidates/."
        )

    results_df = pd.DataFrame(results)

    output_path = METRICS_DIR / "threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    # --------------------------------------------------
    # 3. Best F1 threshold for each model
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("BEST F1 THRESHOLD FOR EACH MODEL")
    print("=" * 65)

    best_f1 = results_df.loc[
        results_df.groupby("model")["f1_score"].idxmax()
    ]

    print(
        best_f1[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "f2_score",
                "flagged_transactions",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 4. High recall with a precision requirement
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("HIGHEST RECALL WITH PRECISION >= 0.50")
    print("=" * 65)

    eligible = results_df[results_df["precision"] >= 0.50]

    if eligible.empty:
        print("No threshold met the precision requirement.")
    else:
        best_recall = eligible.loc[
            eligible.groupby("model")["recall"].idxmax()
        ]

        print(
            best_recall[
                [
                    "model",
                    "threshold",
                    "precision",
                    "recall",
                    "f1_score",
                    "f2_score",
                    "flagged_transactions",
                ]
            ].to_string(index=False)
        )

    print("\nAll threshold results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
