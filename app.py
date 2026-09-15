"""
Minimal Streamlit app — repeat-purchase / churn prediction only.
Reads the expected feature names directly off the fitted pipeline
(feature_names_in_), so no separate schema file is needed.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/churn_model.pkl")
LOG_PATH = Path("logs/prediction_log.csv")
LOG_PATH.parent.mkdir(exist_ok=True)

st.set_page_config(page_title="Repeat Purchase Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def log_prediction(inputs: dict, outputs: dict):
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "inputs_json", "outputs_json"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), json.dumps(inputs), json.dumps(outputs)])


st.title("🔁 Repeat Purchase Predictor")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Make sure `models/churn_model.pkl` is in the repo.")
    st.stop()

model = load_model()
feature_cols = list(model.feature_names_in_)

st.write("Enter a customer's profile to estimate the probability of a repeat purchase.")

input_vals = {}
cols = st.columns(2)
for i, c in enumerate(feature_cols):
    with cols[i % 2]:
        input_vals[c] = st.number_input(c.replace("_", " ").title(), value=0.0, min_value=0.0)

if st.button("Predict", type="primary"):
    X_new = pd.DataFrame([input_vals])[feature_cols]
    proba = float(model.predict_proba(X_new)[0, 1])
    pred = int(proba >= 0.5)

    st.metric("Repeat-purchase probability", f"{proba:.1%}")
    st.write("Predicted class:", "**Repeat purchaser**" if pred else "**One-time / churned**")

    log_prediction(inputs=input_vals, outputs={"probability": proba, "predicted_class": pred})

st.caption(f"Model: Gradient Boosting · features expected: {len(feature_cols)} · logs: `{LOG_PATH}`")
