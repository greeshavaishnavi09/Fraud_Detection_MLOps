from pathlib import Path
import pandas as pd

from ingestion import load_raw_data


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df: pd.DataFrame) -> bool:

    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    validation_passed = True

    # --------------------------------------------------------
    # 1. SHAPE
    # --------------------------------------------------------

    print("\n1. DATASET SHAPE")
    print("-" * 50)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")


    # --------------------------------------------------------
    # 2. COLUMN VALIDATION
    # --------------------------------------------------------

    print("\n2. COLUMN VALIDATION")
    print("-" * 50)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(f"❌ Missing columns: {missing_columns}")
        validation_passed = False

    else:

        print("✅ All expected columns are present")


    # --------------------------------------------------------
    # 3. DUPLICATES
    # --------------------------------------------------------

    print("\n3. DUPLICATE CHECK")
    print("-" * 50)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count:,}")

    if duplicate_count == 0:
        print("✅ No duplicate rows found")
    else:
        print("⚠️ Duplicate rows found")


    # --------------------------------------------------------
    # 4. MISSING VALUES
    # --------------------------------------------------------

    print("\n4. MISSING VALUE CHECK")
    print("-" * 50)

    missing_values = df.isnull().sum()

    total_missing = missing_values.sum()

    print(f"Total missing values: {total_missing:,}")

    if total_missing == 0:

        print("✅ No missing values found")

    else:

        print("\nColumns containing missing values:")

        print(
            missing_values[
                missing_values > 0
            ]
        )

        validation_passed = False


    # --------------------------------------------------------
    # 5. DATA TYPES
    # --------------------------------------------------------

    print("\n5. DATA TYPES")
    print("-" * 50)

    print(df.dtypes)


    # --------------------------------------------------------
    # 6. TRANSACTION TYPES
    # --------------------------------------------------------

    print("\n6. TRANSACTION TYPES")
    print("-" * 50)

    print(df["type"].value_counts())


    # --------------------------------------------------------
    # 7. TARGET VALIDATION
    # --------------------------------------------------------

    print("\n7. TARGET VALIDATION")
    print("-" * 50)

    print("Target column: isFraud")

    print(
        df["isFraud"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # 8. FRAUD DISTRIBUTION
    # --------------------------------------------------------

    print("\n8. FRAUD DISTRIBUTION")
    print("-" * 50)

    fraud_counts = df["isFraud"].value_counts()

    legitimate_count = fraud_counts.get(0, 0)
    fraud_count = fraud_counts.get(1, 0)

    total_transactions = len(df)

    legitimate_percentage = (
        legitimate_count / total_transactions
    ) * 100

    fraud_percentage = (
        fraud_count / total_transactions
    ) * 100

    print(
        f"Legitimate transactions : "
        f"{legitimate_count:,} "
        f"({legitimate_percentage:.4f}%)"
    )

    print(
        f"Fraudulent transactions  : "
        f"{fraud_count:,} "
        f"({fraud_percentage:.4f}%)"
    )


    # --------------------------------------------------------
    # 9. TARGET VALUE CHECK
    # --------------------------------------------------------

    print("\n9. TARGET VALUE CHECK")
    print("-" * 50)

    unique_targets = sorted(
        df["isFraud"].dropna().unique()
    )

    print(f"Unique target values: {unique_targets}")

    if set(unique_targets).issubset({0, 1}):

        print("✅ Target contains valid binary values")

    else:

        print("❌ Unexpected target values found")

        validation_passed = False


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if validation_passed:

        print("✅ DATA VALIDATION PASSED")

    else:

        print("⚠️ DATA VALIDATION COMPLETED WITH ISSUES")

    print("=" * 70)

    return validation_passed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_raw_data()

    validate_data(df)
