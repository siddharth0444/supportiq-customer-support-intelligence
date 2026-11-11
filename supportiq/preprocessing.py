"""Ticket text normalization and CSV validation."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_text(value: object) -> str:
    """Normalize whitespace and URLs while retaining useful ticket wording."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_ticket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a cleaned copy of a dataframe with a text column."""
    if frame.empty:
        raise ValueError("The uploaded file contains no rows.")
    if "text" not in frame.columns:
        raise ValueError("Missing required column: text")
    cleaned = frame.copy()
    cleaned["text"] = cleaned["text"].map(normalize_text)
    cleaned = cleaned.loc[cleaned["text"].ne("")].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No non-empty ticket text was found.")
    return cleaned
