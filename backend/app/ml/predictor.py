
import pandas as pd

from backend.app.ml.model_loader import get_model


# These must match the features used during training.
FEATURES = ["step", "amount", "oldbalanceOrg", "type"]

# Threshold selected using validation data.
FRAUD_THRESHOLD = 0.97


def predict_fraud(transaction: dict) -> dict:
    """
    Predict whether a transaction is fraudulent.

    Required input:
        step, amount, oldbalanceOrg, type
    """

    missing_features = [
        feature for feature in FEATURES
        if feature not in transaction
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Build a single-row dataframe in training feature order.
    transaction_df = pd.DataFrame(
        [{feature: transaction[feature] for feature in FEATURES}],
        columns=FEATURES,
    )

    if transaction_df.isnull().any().any():
        raise ValueError("Transaction features cannot be null.")

    if transaction_df["type"].iloc[0] not in {
        "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
    }:
        raise ValueError("Unsupported transaction type.")

    if transaction_df["step"].iloc[0] < 0:
        raise ValueError("step cannot be negative.")

    if transaction_df["amount"].iloc[0] < 0:
        raise ValueError("amount cannot be negative.")

    if transaction_df["oldbalanceOrg"].iloc[0] < 0:
        raise ValueError("oldbalanceOrg cannot be negative.")

    model = get_model()

    fraud_probability = float(
        model.predict_proba(transaction_df)[0, 1]
    )

    is_fraud = fraud_probability >= FRAUD_THRESHOLD

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": round(fraud_probability, 6),
        "threshold": FRAUD_THRESHOLD,
        "risk_level": (
            "HIGH" if is_fraud else "LOW"
        ),
    }
