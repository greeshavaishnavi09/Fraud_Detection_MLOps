
from fastapi import FastAPI

from backend.app.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent financial transactions.",
    version="1.0.0",
)

app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
