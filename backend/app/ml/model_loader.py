
from pathlib import Path

import joblib


# Project root: Fraud_Detection_MLOps/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "fraud_xgboost.joblib"
)


def load_model():
    """Load and return the approved fraud detection model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    return model


# Lazy loading: load only when requested.
_model = None


def get_model():
    """Return the cached model, loading it on first use."""

    global _model

    if _model is None:
        _model = load_model()

    return _model
