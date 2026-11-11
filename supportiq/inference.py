"""Load the intent model and create structured ticket predictions."""
from __future__ import annotations

import joblib

from supportiq.config import MODEL_PATH
from supportiq.preprocessing import normalize_text
from supportiq.rules import analyze_sentiment, predict_priority


class ModelNotTrainedError(RuntimeError):
    """Raised when prediction is requested before the model is trained."""


class TicketPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline = joblib.load(model_path) if model_path.exists() else None

    def predict(self, text: str) -> dict:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("Ticket text cannot be empty.")
        if self.pipeline is None:
            raise ModelNotTrainedError(
                "Intent model is missing. Run python scripts/generate_demo_data.py "
                "and python -m supportiq.train first."
            )
        category = str(self.pipeline.predict([normalized])[0])
        sentiment = analyze_sentiment(normalized)
        confidence = None
        if hasattr(self.pipeline, "predict_proba"):
            confidence = float(max(self.pipeline.predict_proba([normalized])[0]))
        return {
            "category": category,
            "sentiment": sentiment,
            "priority": predict_priority(normalized, sentiment),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "model_version": "tfidf-logreg-v1",
        }
