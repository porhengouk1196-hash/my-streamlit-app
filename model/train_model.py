"""
Trains all 4 candidate models and saves them to model/credit_risk_model.pkl.

Run this once before starting the Streamlit app:

    python model/train_model.py

Why this script exists (instead of shipping a pre-trained .pkl): this
project was assembled in an environment without scikit-learn / internet
access, so the models could not be trained and pickled there. This script
reproduces the exact cleaning + encoding + training pipeline documented in
the source notebook.

All 4 models are kept in the saved bundle (not just the best one) so the
app can let you switch between them and compare predicted scores.

Expects cr_loan.csv in the project root (same folder as this script's
parent directory) unless --data is passed explicitly.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import clean_dataset, encode_features  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RANDOM_STATE = 42


def evaluate(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob),
    }


def feature_importance_for(model, columns):
    """Tree models expose feature_importances_ directly; Logistic Regression
    uses the absolute value of its coefficients as a comparable (if less
    precise) measure of feature influence."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(columns))
    return pd.Series(importances, index=columns).sort_values(ascending=False).head(10).to_dict()


def main(data_path: str, output_path: str) -> None:
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Raw shape: {df.shape}")

    df_clean = clean_dataset(df)
    print(f"Cleaned shape: {df_clean.shape}")

    df_model = encode_features(df_clean)

    X = df_model.drop(columns=["loan_status"])
    y = df_model["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Logistic Regression needs scaled features; tree-based models do not.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    fitted_models = {}
    confusion_matrices = {}
    feature_importances = {}

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    lr_pred = log_reg.predict(X_test_scaled)
    results.append(evaluate("Logistic Regression", y_test, lr_pred, log_reg.predict_proba(X_test_scaled)[:, 1]))
    fitted_models["Logistic Regression"] = log_reg
    confusion_matrices["Logistic Regression"] = confusion_matrix(y_test, lr_pred).tolist()
    feature_importances["Logistic Regression"] = feature_importance_for(log_reg, X.columns)

    dec_tree = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    dec_tree.fit(X_train, y_train)
    dt_pred = dec_tree.predict(X_test)
    results.append(evaluate("Decision Tree", y_test, dt_pred, dec_tree.predict_proba(X_test)[:, 1]))
    fitted_models["Decision Tree"] = dec_tree
    confusion_matrices["Decision Tree"] = confusion_matrix(y_test, dt_pred).tolist()
    feature_importances["Decision Tree"] = feature_importance_for(dec_tree, X.columns)

    rand_forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rand_forest.fit(X_train, y_train)
    rf_pred = rand_forest.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred, rand_forest.predict_proba(X_test)[:, 1]))
    fitted_models["Random Forest"] = rand_forest
    confusion_matrices["Random Forest"] = confusion_matrix(y_test, rf_pred).tolist()
    feature_importances["Random Forest"] = feature_importance_for(rand_forest, X.columns)

    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_model.predict_proba(X_test)[:, 1]))
        fitted_models["XGBoost"] = xgb_model
        confusion_matrices["XGBoost"] = confusion_matrix(y_test, xgb_pred).tolist()
        feature_importances["XGBoost"] = feature_importance_for(xgb_model, X.columns)
    else:
        print("xgboost not installed — skipping (3 models will still be saved).")

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    results_df = results_df.sort_values("ROC AUC", ascending=False)
    print("\nModel comparison (sorted by ROC AUC):")
    print(results_df)

    best_model_name = results_df.index[0]
    print(f"\nBest model by ROC AUC: {best_model_name}")

    bundle = {
        "models": fitted_models,               # every trained model, keyed by name
        "columns": list(X.columns),
        "best_model_name": best_model_name,
        "scaler": scaler,                       # only used for Logistic Regression at inference time
        "metrics": results_df.to_dict(orient="index"),
        "confusion_matrix": confusion_matrices,
        "feature_importance": feature_importances,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"\nSaved model bundle ({len(fitted_models)} models) to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit risk models.")
    default_data = os.path.join(os.path.dirname(__file__), "..", "cr_loan.csv")
    default_output = os.path.join(os.path.dirname(__file__), "credit_risk_model.pkl")
    parser.add_argument("--data", default=default_data, help="Path to cr_loan.csv")
    parser.add_argument("--output", default=default_output, help="Where to save the trained model bundle")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: dataset not found at '{args.data}'. "
              f"Place cr_loan.csv in the project root, or pass --data <path>.")
        sys.exit(1)

    main(args.data, args.output)
