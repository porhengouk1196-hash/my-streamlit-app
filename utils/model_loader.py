"""
Loads the trained model bundle saved by model/train_model.py.

The bundle is a single dict so every model, the exact column layout it was
trained on, and comparison metrics all travel together:
    {
        "models": {"Logistic Regression": ..., "Decision Tree": ...,
                   "Random Forest": ..., "XGBoost": ...},
        "columns": [<feature names in training order>],
        "best_model_name": "XGBoost",
        "scaler": <fitted StandardScaler, used only for Logistic Regression>,
        "metrics": {<model name>: {"Accuracy": ..., "ROC AUC": ..., ...}},
        "confusion_matrix": {<model name>: [[..],[..]]},
        "feature_importance": {<model name>: {<feature>: <importance>}},
    }
"""

import os
import streamlit as st
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "credit_risk_model.pkl")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Load the model bundle once per session (cached). Returns None if the
    file doesn't exist yet, so callers can show a friendly setup message
    instead of a stack trace."""
    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
        st.error(f"Failed to load model from '{path}': {exc}")
        return None
