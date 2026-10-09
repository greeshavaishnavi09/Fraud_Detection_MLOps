from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# RAW DATA PATH
# ============================================================

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "PS_20174392719_1491204439457_log.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data() -> pd.DataFrame:

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_raw_data()

    print("=" * 70)
    print("DATA INGESTION SUCCESSFUL")
    print("=" * 70)

    print(f"\nDataset path:")
    print(RAW_DATA_PATH)

    print(f"\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())