
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "metrics"

TARGET = "isFraud"

NUMERICAL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
]

CATEGORICAL_FEATURES = ["type"]


def build_preprocessor():

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
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


if __name__ == "__main__":

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")

    X_train = train_df[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ]

    preprocessor = build_preprocessor()

    # Fit only on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        ARTIFACT_DIR / "preprocessor.joblib",
    )

    print("Preprocessing pipeline fitted successfully.")
    print("Original feature count:", X_train.shape[1])
    print(
        "Transformed feature count:",
        X_train_transformed.shape[1],
    )
    print("Training rows:", X_train.shape[0])
    print("Preprocessor saved.")
