"""FastAPI endpoints for individual and batch ticket analysis."""
from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from supportiq.inference import ModelNotTrainedError, TicketPredictor
from supportiq.preprocessing import validate_ticket_frame

app = FastAPI(
    title="SupportIQ API",
    description="Baseline intent classification, sentiment and priority triage.",
    version="0.1.0",
)
predictor = TicketPredictor()


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": predictor.pipeline is not None}


@app.post("/predict")
def predict_ticket(request: TicketRequest) -> dict:
    try:
        return predictor.predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    raw = await file.read()
    if len(raw) > 5_000_000:
        raise HTTPException(status_code=413, detail="CSV must be smaller than 5 MB.")
    try:
        frame = validate_ticket_frame(pd.read_csv(BytesIO(raw)))
        results = []
        for index, row in frame.iterrows():
            results.append({
                "row": int(index),
                "text": row["text"],
                **predictor.predict(row["text"]),
            })
        counts = pd.DataFrame(results)["category"].value_counts().to_dict()
        return {"total_tickets": len(results), "category_counts": counts, "tickets": results}
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc
