# SupportIQ — Customer Support Intelligence

SupportIQ is an open-source customer-support analytics MVP. It classifies ticket intent, estimates sentiment, flags urgency, and makes ticket trends explorable.

> **Status:** MVP foundation. The baseline is trained on synthetic demo data. Metrics from synthetic data demonstrate the pipeline only; they are not production-performance claims.

## Features
- CSV ticket ingestion and validation
- Intent classification with TF-IDF + Logistic Regression
- Transparent lexicon sentiment baseline
- Explainable rule-based priority triage
- FastAPI prediction and batch-analysis endpoints
- Streamlit dashboard with ticket analytics
- Reproducible demo data, tests, linting and GitHub Actions CI

## Quick start
Python 3.10+ recommended.

```bash
git clone https://github.com/siddharth0444/supportiq-customer-support-intelligence.git
cd supportiq-customer-support-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/generate_demo_data.py
python -m supportiq.train
```

Run in separate terminals:

```bash
uvicorn supportiq.api:app --reload
streamlit run app.py
```

Dashboard: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs

## API
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"I was charged twice and need a refund immediately"}'
```

Batch endpoint: `POST /analyze-csv` with multipart field `file` containing a CSV with a `text` column.

## Repository layout
```text
app.py
supportiq/       # preprocessing, rules, training, inference and API
scripts/         # synthetic demo data generator
tests/           # API, preprocessing and rules tests
data/            # generated sample tickets
.github/workflows/ci.yml
```

## Data and model limitations
The generator creates deterministic synthetic examples. These are illustrative rather than representative of real customer language. Sentiment is a small keyword lexicon and does not understand sarcasm, context or multilingual text. Priority is a transparent heuristic, not a calibrated risk model.

For real use, replace generated data with permissioned, de-identified tickets, create human-reviewed labels, compare multiple models, evaluate per-class precision/recall and calibration, and monitor drift. Keep a human in the loop for high-impact decisions.

Do not upload confidential or sensitive customer information to a public demo. Never use this MVP to deny support or customer rights.

## Development
```bash
pytest
ruff check .
```

## Roadmap
- [x] Reproducible project structure and synthetic demo data
- [x] Intent classification baseline
- [x] Sentiment and priority baselines
- [x] Prediction API and Streamlit MVP
- [x] Tests and CI workflow
- [ ] Human-labelled benchmark and model comparison
- [ ] Database-backed ticket history and authentication
- [ ] Monitoring, drift reports and deployment

## License
MIT.
