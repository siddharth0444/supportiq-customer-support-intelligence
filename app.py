"""Streamlit dashboard for SupportIQ."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix

from supportiq.config import DEMO_DATA_PATH
from supportiq.inference import ModelNotTrainedError, TicketPredictor

st.set_page_config(page_title="SupportIQ", page_icon="🎧", layout="wide")
st.title("🎧 SupportIQ")
st.caption("Customer Support Intelligence · NLP triage and ticket analytics")


@st.cache_resource
def load_predictor():
    return TicketPredictor()


predictor = load_predictor()
with st.sidebar:
    st.header("Workspace")
    st.info("MVP demo mode · Synthetic tickets")
    st.caption("Do not upload confidential customer information.")

st.subheader("Analyze a ticket")
ticket_text = st.text_area(
    "Customer message",
    placeholder="I was charged twice and need a refund immediately.",
    height=120,
)
if st.button("Analyze ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Enter a customer message first.")
    else:
        try:
            result = predictor.predict(ticket_text)
            cols = st.columns(4)
            cols[0].metric("Intent", result["category"].title())
            cols[1].metric("Sentiment", result["sentiment"].title())
            cols[2].metric("Priority", result["priority"].upper())
            confidence = result["confidence"]
            cols[3].metric("Confidence", f"{confidence:.1%}" if confidence is not None else "N/A")
        except ModelNotTrainedError as exc:
            st.error(str(exc))

st.divider()
st.subheader("Ticket explorer")
uploaded = st.file_uploader("Upload CSV with a text column (optional)", type=["csv"])
try:
    frame = pd.read_csv(uploaded) if uploaded else pd.read_csv(DEMO_DATA_PATH)
    if "text" not in frame.columns:
        st.error("CSV must include a text column.")
    else:
        frame["text"] = frame["text"].fillna("").astype(str)
        if st.button("Classify all tickets"):
            try:
                valid = frame.loc[frame["text"].str.strip().ne("")].reset_index(drop=True)
                predictions = [predictor.predict(value) for value in valid["text"]]
                analyzed = pd.DataFrame(predictions).rename(
                    columns={"category": "predicted_category"})
                cols = st.columns(3)
                cols[0].metric("Analyzed tickets", len(analyzed))
                cols[1].metric("High priority", int((analyzed["priority"] == "high").sum()))
                cols[2].metric("Negative sentiment", int((analyzed["sentiment"] == "negative").sum()))
                st.plotly_chart(
                    px.histogram(analyzed, x="predicted_category", color="priority", barmode="group",
                    title="Ticket volume by predicted category"),
                    use_container_width=True,
                )
                
                # Evaluate predictions against the original dataset labels
                if "category" in valid.columns:
                    actual = valid["category"].astype(str).str.lower()
                    predicted = analyzed["predicted_category"].astype(str).str.lower()

                    accuracy = accuracy_score(actual, predicted)

                    st.subheader("Model evaluation")
                    st.metric("Classification accuracy", f"{accuracy:.1%}")

                    labels = sorted(set(actual) | set(predicted))
                    matrix = confusion_matrix(actual, predicted, labels=labels)

                    fig = px.imshow(
                        matrix,
                        x=labels,
                        y=labels,
                        text_auto=True,
                        labels={
                            "x": "Predicted category",
                            "y": "Actual category",
                            "color": "Ticket count",
                        },
                        title="Confusion matrix",
                        color_continuous_scale="Blues",
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No original category labels available for evaluation.")

                st.dataframe(
                    pd.concat([valid, analyzed], axis=1),
                    use_container_width=True,
                )

            except ModelNotTrainedError as exc:
                st.error(str(exc))
        else:
            st.caption(f"{len(frame):,} rows available. Select Classify all tickets to analyze.")
except FileNotFoundError:
    st.warning("Demo data not found. Run scripts/generate_demo_data.py.")
