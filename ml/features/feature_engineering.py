
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_cleaned.csv"
)

OUTPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_features.csv"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction features for the fraud detection project.

    The original target column is retained.
    The input DataFrame is not modified.
    """

    df = df.copy()

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # --------------------------------------------------------
    # 1. Sender balance change
    # --------------------------------------------------------

    df["origin_balance_change"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # --------------------------------------------------------
    # 2. Receiver balance change
    # --------------------------------------------------------

    df["destination_balance_change"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # --------------------------------------------------------
    # 3. Sender balance discrepancy
    # --------------------------------------------------------

    df["origin_balance_error"] = (
        df["origin_balance_change"] - df["amount"]
    )

    # --------------------------------------------------------
    # 4. Receiver balance discrepancy
    # --------------------------------------------------------

    df["destination_balance_error"] = (
        df["destination_balance_change"] - df["amount"]
    )

    # --------------------------------------------------------
    # 5. Transaction amount relative to sender balance
    # --------------------------------------------------------

    df["amount_to_origin_balance_ratio"] = (
        df["amount"]
        / df["oldbalanceOrg"].replace(0, np.nan)
    )

    # Avoid infinite ratios and preserve missing values
    df["amount_to_origin_balance_ratio"] = (
        df["amount_to_origin_balance_ratio"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # 6. Log-transformed transaction amount
    # --------------------------------------------------------

    df["log_amount"] = np.log1p(df["amount"])

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not INPUT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_DATA_PATH}"
        )

    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_DATA_PATH)

    print("Original shape:", df.shape)

    print("\nCreating features...")

    feature_df = create_features(df)

    print("\nFeature engineering completed.")
    print("New shape:", feature_df.shape)

    new_features = [
        "origin_balance_change",
        "destination_balance_change",
        "origin_balance_error",
        "destination_balance_error",
        "amount_to_origin_balance_ratio",
        "log_amount",
    ]

    print("\nCreated features:")
    print(new_features)

    print("\nFeature preview:")
    print(feature_df[new_features].head())

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
    )

    print("\nFeatures saved to:")
    print(OUTPUT_DATA_PATH)
