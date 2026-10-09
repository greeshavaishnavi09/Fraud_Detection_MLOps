
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TARGET_COLUMN = "isFraud"
RANDOM_STATE = 42


# ============================================================
# DATA SPLITTING
# ============================================================

def split_data(df: pd.DataFrame):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Target contains missing values.")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    # Step 1: Separate features and target
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    # Step 2: Split off 15% test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Step 3: Split the remaining data into train and validation.
    # 15% of the original dataset is 15/85 of this remainder.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading engineered dataset...")

    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save each feature set with its target attached
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    val_df = X_val.copy()
    val_df[TARGET_COLUMN] = y_val

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSPLIT SUMMARY")
    print("=" * 60)

    for name, features, target in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        fraud_count = int((target == 1).sum())

        print(f"\n{name}")
        print(f"Rows: {len(features):,}")
        print(f"Fraud: {fraud_count:,}")
        print(f"Fraud rate: {target.mean() * 100:.4f}%")

    print("\nAll splits saved to:", OUTPUT_DIR)
