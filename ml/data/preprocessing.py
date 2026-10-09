
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make ingestion.py importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paysim_cleaned.csv"
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect and clean the dataset without modifying raw data."""

    df = df.copy()

    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)

    print(f"Original shape: {df.shape}")

    # 1. Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 2. Duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())

    # 3. Infinite numerical values
    numeric_df = df.select_dtypes(include=np.number)
    infinite_count = np.isinf(numeric_df.to_numpy()).sum()
    print("\nInfinite numerical values:", int(infinite_count))

    # 4. Target validation
    if "isFraud" not in df.columns:
        raise ValueError("Required target column 'isFraud' is missing.")

    if df["isFraud"].isna().any():
        raise ValueError("Missing fraud labels must be investigated.")

    if not set(df["isFraud"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected values in isFraud.")

    # 5. Report suspicious negative values
    check_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for column in check_columns:
        if column in df.columns:
            print(
                f"Negative values in {column}: "
                f"{int((df[column] < 0).sum())}"
            )

    # Do not silently drop duplicates or anomalous transactions.
    # Stop if issues require a cleaning decision.
    if df.isnull().any().any():
        raise ValueError("Investigate missing values before continuing.")

    if infinite_count > 0:
        raise ValueError("Investigate infinite values before continuing.")

    print("\nNo rows were automatically removed.")
    print(f"Shape after checks: {df.shape}")

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed data saved to: {PROCESSED_DATA_PATH}")
