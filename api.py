"""FastAPI application for spam prediction."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import re
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

app = FastAPI(title="Text Spam Detection API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request schema for spam prediction."""

    text: str = Field(..., min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject empty or whitespace-only messages."""
        if not value or not value.strip():
            raise ValueError("Text must not be empty or whitespace only.")
        if len(value) > 1000:
            raise ValueError("Text must be at most 1000 characters long.")
        return value.strip()


class PredictionResponse(BaseModel):
    """Response schema for spam prediction."""

    text: str
    prediction: str
    confidence: float
    algorithm: str


model: MultinomialNB | None = None
vectorizer: Any | None = None


def clean_text(text: str) -> str:
    """Normalize incoming SMS text using identical training rules for robust inference."""
    if not isinstance(text, str):
        return ""
    
    # 1. Standardize case early
    text = text.lower()
    
    # 2. Map URLs and numerical indicators to generic variables before character stripping
    text = re.sub(r'http\S+|www\S+', 'urltoken', text)
    text = re.sub(r'\d+', 'numbertoken', text)
    
    # 3. Collapse aggressive token separation and punctuation padding
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Standardize space blocks
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.on_event("startup")
def load_model() -> None:
    """Load the trained model and vectorizer at startup."""
    global model, vectorizer
    logger.info("API startup: loading model artifacts")
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        logger.error("Model files not found")
        raise FileNotFoundError("Model files are missing. Please run train.py first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("API startup: model artifacts loaded successfully")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health status response."""
    return {"status": "running"}


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Serve the browser-based spam checker frontend."""
    index_path = PROJECT_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend page not found.")
    return FileResponse(index_path, media_type="text/html")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict whether the input message is spam or ham with a tuned sensitivity threshold."""
    logger.info("Prediction request received for text: %s", request.text)

    if model is None or vectorizer is None:
        logger.error("Prediction attempted before model loading")
        raise HTTPException(status_code=503, detail="Model is not available yet.")

    try:
        # Preprocess the incoming request string using the unified cleaning rules
        cleaned_text = clean_text(request.text)
        
        # Extract features using the loaded character-level TF-IDF vectorizer
        transformed_text = vectorizer.transform([cleaned_text])
        probability = model.predict_proba(transformed_text)[0]
        
        # Extract probabilities explicitly: index 0 is HAM, index 1 is SPAM
        ham_probability = probability[0]
        spam_probability = probability[1]
        
        # Lower the threshold barrier to bypass dataset class imbalance skew
        # If the model is even 10% sure it's spam, we flag it.
        if spam_probability > 0.10:
            predicted_label = "SPAM"
            confidence = float(spam_probability * 100)
        else:
            predicted_label = "HAM"
            confidence = float(ham_probability * 100)

        return PredictionResponse(
            text=request.text,
            prediction=predicted_label,
            confidence=round(confidence, 2),
            algorithm="Optimized Multinomial Naive Bayes",
        )
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed due to an unexpected error.") from exc

@app.exception_handler(RequestValidationError)
def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return meaningful validation errors for bad requests."""
    logger.warning("Validation error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Return consistent JSON for HTTP errors."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions gracefully."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)