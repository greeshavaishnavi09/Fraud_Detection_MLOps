
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

TARGET = "isFraud"
THRESHOLD = 0.5

NUMERICAL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
]

CATEGORICAL_FEATURES = ["type"]


# ============================================================
# LOAD TRAINING AND VALIDATION DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "validation.csv")

    X_train = train_df[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ]
    y_train = train_df[TARGET]

    X_val = val_df[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ]
    y_val = val_df[TARGET]

    return X_train, y_train, X_val, y_val


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_model():

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = {
        "model": "Logistic Regression",
        "threshold": THRESHOLD,
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_val, probabilities
        ),
    }

    print("\nVALIDATION RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    X_train, y_train, X_val, y_val = load_data()

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))

    print("\nBuilding Logistic Regression pipeline...")

    model = build_model()

    print("\nTraining baseline model...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val)

    model_path = MODEL_DIR / "baseline_logistic_regression.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / "baseline_metrics.csv",
        index=False,
    )

    print("\nModel saved to:", model_path)
    print("Metrics saved to:", METRICS_DIR / "baseline_metrics.csv")
