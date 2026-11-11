"""Train and evaluate the baseline intent classifier."""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from supportiq.config import ARTIFACTS_DIR, DEMO_DATA_PATH, MODEL_PATH
from supportiq.preprocessing import validate_ticket_frame


def train_model(data_path=DEMO_DATA_PATH) -> dict:
    frame = validate_ticket_frame(pd.read_csv(data_path))
    if "category" not in frame.columns:
        raise ValueError("Training data must contain a category column.")
    if frame["category"].nunique() < 2:
        raise ValueError("Training requires at least two categories.")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"], frame["category"], test_size=0.25,
        random_state=42, stratify=frame["category"],
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    synthetic_demo_data = bool(
        "source" in frame.columns and frame["source"].eq("synthetic_demo").all()
    )
    metrics = {
        "rows": int(len(frame)),
        "synthetic_demo_data": synthetic_demo_data,
        "model": "TF-IDF + Logistic Regression",
        "classification_report": report,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_test, predictions, zero_division=0))
    return metrics


if __name__ == "__main__":
    train_model()
