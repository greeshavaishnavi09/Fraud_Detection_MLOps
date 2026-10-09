
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
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

from xgboost import XGBClassifier


# 1. Paths and configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
)
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


def main():
    start_time = time.time()

    # 2. Load data

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)

    print("Loading validation data...")
    validation_df = pd.read_csv(VALIDATION_PATH)

    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("training", train_df),
        ("validation", validation_df),
    ]:
        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {sorted(missing)}"
            )

    print("Original training shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)

    # 3. Stratified training sample

    if len(train_df) > TRAIN_SAMPLE_SIZE:
        train_df, _ = train_test_split(
            train_df,
            train_size=TRAIN_SAMPLE_SIZE,
            stratify=train_df[TARGET],
            random_state=RANDOM_STATE,
        )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET].astype(int)

    del train_df
    del validation_df

    print("\nSampled training shape:", X_train.shape)
    print("Training class counts:")
    print(y_train.value_counts())

    # 4. Calculate class imbalance weight from training data only

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    # 5. Preprocessing

    numerical_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
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

    # 6. XGBoost classifier

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # 7. Train

    print("\nTraining XGBoost...")
    print("This may take several minutes.")

    model.fit(X_train, y_train)

    print("Training completed.")

    # 8. Validation predictions

    print("\nGenerating validation predictions...")

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # 9. Evaluation

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
    print("XGBOOST - VALIDATION RESULTS")
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

    # 10. Save model and metrics

    model_path = MODEL_DIR / "xgboost.joblib"
    metrics_path = METRICS_DIR / "xgboost_metrics.csv"

    joblib.dump(model, model_path)

    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost",
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
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
