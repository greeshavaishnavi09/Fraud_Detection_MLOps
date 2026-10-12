
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.predictor import predict_fraud

router = APIRouter(prefix="/api/v1", tags=["Fraud Prediction"])


class TransactionRequest(BaseModel):
    step: int = Field(ge=0)
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    type: str


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold: float
    risk_level: str


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(transaction: TransactionRequest):
    try:
        return predict_fraud(transaction.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
