"""Shared application paths."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "intent_pipeline.joblib"
DEMO_DATA_PATH = DATA_DIR / "sample_tickets.csv"
