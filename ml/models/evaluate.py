
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "candidates" / "xgboost.joblib"
)

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
FEATURES = ["step", "amount", "oldbalanceOrg", "type"]

# Selected using validation data; do not tune on the test set.
THRESHOLD = 0.97


# --------------------------------------------------
# 2. Load test data and selected model
# --------------------------------------------------

def main():
    print("Loading held-out test dataset...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    print("Test dataset shape:", test_df.shape)
    print("Test class distribution:")
    print(y_test.value_counts())

    print("\nLoading selected XGBoost model...")
    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # 3. Generate predictions
    # --------------------------------------------------

    print(f"\nUsing fixed threshold: {THRESHOLD}")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # --------------------------------------------------
    # 4. Calculate evaluation metrics
    # --------------------------------------------------

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    f2 = fbeta_score(
        y_test, predictions, beta=2, zero_division=0
    )
    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)

    tn, fp, fn, tp = matrix.ravel()

    # --------------------------------------------------
    # 5. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL XGBOOST TEST RESULTS")
    print("=" * 60)

    print(f"Threshold:  {THRESHOLD:.2f}")
    print(f"Precision:  {precision:.6f}")
    print(f"Recall:     {recall:.6f}")
    print(f"F1-score:   {f1:.6f}")
    print(f"F2-score:   {f2:.6f}")
    print(f"PR-AUC:     {pr_auc:.6f}")
    print(f"ROC-AUC:    {roc_auc:.6f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("\nConfusion matrix interpretation:")
    print(f"True negatives  (legitimate correctly identified): {tn}")
    print(f"False positives (legitimate flagged as fraud):     {fp}")
    print(f"False negatives (fraud missed):                    {fn}")
    print(f"True positives  (fraud correctly identified):      {tp}")

    # --------------------------------------------------
    # 6. Save test metrics
    # --------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "XGBoost",
            "evaluation_dataset": "test",
            "threshold": THRESHOLD,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f2_score": f2,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    ])

    output_path = METRICS_DIR / "final_test_metrics.csv"
    results.to_csv(output_path, index=False)

    print("\nTest metrics saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
