
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

MODEL_DIR = PROJECT_ROOT / "models" / "candidates"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load configuration
# --------------------------------------------------

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# --------------------------------------------------
# 3. Training pipeline
# --------------------------------------------------

def main():
    print("=" * 60)
    print("FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    config = load_config()

    target = config["data"]["target_column"]

    train_path = PROJECT_ROOT / config["data"]["train_path"]

    numerical_features = config["features"]["numerical"]
    categorical_features = config["features"]["categorical"]

    features = numerical_features + categorical_features

    random_state = config["project"]["random_state"]
    sample_size = config["training"]["sample_size"]

    print("\nLoading training dataset...")
    train_df = pd.read_csv(train_path)

    required_columns = features + [target]
    missing_columns = set(required_columns) - set(train_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("Original training shape:", train_df.shape)

    # --------------------------------------------------
    # 4. Stratified sampling
    # --------------------------------------------------

    if len(train_df) > sample_size:
        train_df, _ = train_test_split(
            train_df,
            train_size=sample_size,
            stratify=train_df[target],
            random_state=random_state,
        )

    X_train = train_df[features]
    y_train = train_df[target].astype(int)

    print("Training sample shape:", X_train.shape)
    print("Training class counts:")
    print(y_train.value_counts())

    # --------------------------------------------------
    # 5. Preprocessing
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
                numerical_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------
    # 6. Configure XGBoost from YAML
    # --------------------------------------------------

    model_config = config["models"]["xgboost"]

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training sample contains no fraud cases.")

    scale_pos_weight = negative_count / positive_count

    classifier_params = {
        "n_estimators": model_config["n_estimators"],
        "max_depth": model_config["max_depth"],
        "learning_rate": model_config["learning_rate"],
        "subsample": model_config["subsample"],
        "colsample_bytree": model_config["colsample_bytree"],
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": model_config["eval_metric"],
        "tree_method": model_config["tree_method"],
        "n_jobs": model_config["n_jobs"],
        "random_state": random_state,
    }

    classifier = XGBClassifier(**classifier_params)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # --------------------------------------------------
    # 7. Train model
    # --------------------------------------------------

    print("\nTraining XGBoost...")
    print("Using parameters:", classifier_params)

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # --------------------------------------------------
    # 8. Save candidate model
    # --------------------------------------------------

    model_path = MODEL_DIR / "xgboost_configured.joblib"

    joblib.dump(model, model_path)

    print("\nCandidate model saved to:")
    print(model_path)

    print("\nTraining pipeline completed.")


if __name__ == "__main__":
    main()
