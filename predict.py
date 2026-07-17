"""Command-line utility to predict whether a message is spam or ham."""

from __future__ import annotations

import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "spam_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "model" / "vectorizer.pkl"


def predict_message(text: str) -> tuple[str, float]:
    """Predict the class and confidence score for a message."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files are missing. Please run train.py first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    transformed_text = vectorizer.transform([text])
    probability = model.predict_proba(transformed_text)[0]
    prediction_index = int(probability.argmax())
    predicted_label = "SPAM" if prediction_index == 1 else "HAM"
    confidence = float(max(probability) * 100)
    return predicted_label, round(confidence, 2)


if __name__ == "__main__":
    message = input("Enter a message: ").strip()
    prediction, confidence = predict_message(message)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
