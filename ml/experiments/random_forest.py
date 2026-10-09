
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "validation.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
NUMERICAL_FEATURES = ["step", "amount", "oldbalanceOrg"]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TRAIN_SAMPLE_SIZE = 500_000
RANDOM_STATE = 42
THRESHOLD = 0.5


# --------------------------------------------------
# 2. Load training and validation datasets
# --------------------------------------------------

def main():
    start_time = time.time()

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading validation data...")
    validation_df = pd.read_csv(VALIDATION_PATH)

    print("\nOriginal training shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)

    # Check required columns
    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("training", train_df),
        ("validation", validation_df),
    ]:
        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name} dataset: {sorted(missing)}"
            )

    # --------------------------------------------------
    # 3. Stratified sampling
    # --------------------------------------------------

    if len(train_df) > TRAIN_SAMPLE_SIZE:
        train_df, _ = train_test_split(
            train_df,
            train_size=TRAIN_SAMPLE_SIZE,
            stratify=train_df[TARGET],
            random_state=RANDOM_STATE,
        )

    print("\nSampled training shape:", train_df.shape)
    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts())
    print(
        "\nTraining fraud percentage:",
        round(train_df[TARGET].mean() * 100, 4),
        "%",
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET].astype(int)

    # Release the original dataframe references
    del train_df
    del validation_df

    # --------------------------------------------------
    # 4. Preprocessing
    # --------------------------------------------------

    numerical_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------
    # 5. Random Forest model
    # --------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", random_forest),
        ]
    )

    # --------------------------------------------------
    # 6. Train
    # --------------------------------------------------

    print("\nTraining Random Forest...")
    print("This may take several minutes.")

    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------
    # 7. Validation predictions
    # --------------------------------------------------

    print("\nGenerating validation predictions...")

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # --------------------------------------------------
    # 8. Evaluation
    # --------------------------------------------------

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    pr_auc = average_precision_score(y_val, probabilities)

    matrix = confusion_matrix(y_val, predictions)

    print("\n" + "=" * 55)
    print("RANDOM FOREST - VALIDATION RESULTS")
    print("=" * 55)

    print(f"Threshold:  {THRESHOLD}")
    print(f"Precision:  {precision:.6f}")
    print(f"Recall:     {recall:.6f}")
    print(f"F1-score:   {f1:.6f}")
    print(f"PR-AUC:     {pr_auc:.6f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 9. Save model and metrics
    # --------------------------------------------------

    model_path = MODEL_DIR / "random_forest.joblib"
    metrics_path = METRICS_DIR / "random_forest_metrics.csv"

    joblib.dump(model, model_path)

    metrics = pd.DataFrame(
        [
            {
                "model": "RandomForestClassifier",
                "training_sample_size": len(X_train),
                "validation_size": len(X_val),
                "threshold": THRESHOLD,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "pr_auc": pr_auc,
                "true_negatives": int(matrix[0, 0]),
                "false_positives": int(matrix[0, 1]),
                "false_negatives": int(matrix[1, 0]),
                "true_positives": int(matrix[1, 1]),
                "training_time_seconds": round(
                    time.time() - start_time, 2
                ),
            }
        ]
    )

    metrics.to_csv(metrics_path, index=False)

    print("\nModel saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print(
        f"\nTotal execution time: "
        f"{time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
