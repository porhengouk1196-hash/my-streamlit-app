"""
Feature preprocessing shared between model/train_model.py (training time)
and pages/prediction.py (inference time).

Keeping this logic in one place is deliberate: if training and inference
encode categorical features differently, the model will silently produce
wrong predictions. Both call sites import the SAME functions from here.

Encoding matches the source notebook exactly:
  - loan_grade                  -> ordinal encoding, A=0 ... G=6
  - cb_person_default_on_file   -> binary map, Y=1 / N=0
  - person_home_ownership       -> one-hot encoding (drop_first=True)
  - loan_intent                 -> one-hot encoding (drop_first=True)
"""

import numpy as np
import pandas as pd

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
DEFAULT_FLAG_MAP = {"Y": 1, "N": 0}

HOME_OWNERSHIP_OPTIONS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
LOAN_INTENT_OPTIONS = [
    "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
    "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
]
LOAN_GRADE_OPTIONS = list(GRADE_ORDER.keys())

# Real average interest rate per (loan grade, loan intent) combination,
# computed directly from cr_loan.csv via groupby. Grade is still the
# dominant driver (13+ points end to end, A ~7% to G ~20%), while intent
# only shifts the rate within a grade by a fraction of a point to ~1 point
# at most (real, not fabricated) — so switching Loan Intent now visibly
# moves the auto-filled rate, without pretending intent matters more than
# the data shows it does.
GRADE_INTENT_INTEREST_RATE = {
    "A": {"PERSONAL": 7.33, "EDUCATION": 7.37, "MEDICAL": 7.34, "VENTURE": 7.31, "HOMEIMPROVEMENT": 7.26, "DEBTCONSOLIDATION": 7.33},
    "B": {"PERSONAL": 11.00, "EDUCATION": 11.00, "MEDICAL": 10.99, "VENTURE": 11.01, "HOMEIMPROVEMENT": 10.99, "DEBTCONSOLIDATION": 10.99},
    "C": {"PERSONAL": 13.48, "EDUCATION": 13.43, "MEDICAL": 13.47, "VENTURE": 13.50, "HOMEIMPROVEMENT": 13.44, "DEBTCONSOLIDATION": 13.46},
    "D": {"PERSONAL": 15.37, "EDUCATION": 15.31, "MEDICAL": 15.41, "VENTURE": 15.40, "HOMEIMPROVEMENT": 15.27, "DEBTCONSOLIDATION": 15.39},
    "E": {"PERSONAL": 16.99, "EDUCATION": 17.02, "MEDICAL": 17.08, "VENTURE": 16.78, "HOMEIMPROVEMENT": 17.24, "DEBTCONSOLIDATION": 17.01},
    "F": {"PERSONAL": 19.25, "EDUCATION": 18.40, "MEDICAL": 18.56, "VENTURE": 18.78, "HOMEIMPROVEMENT": 18.21, "DEBTCONSOLIDATION": 18.57},
    "G": {"PERSONAL": 19.89, "EDUCATION": 20.08, "MEDICAL": 20.29, "VENTURE": 20.66, "HOMEIMPROVEMENT": 20.17, "DEBTCONSOLIDATION": 20.33},
}

# Grade-only fallback (simple average across intents), kept as a safety net
# in case an unrecognized intent value ever reaches this function.
GRADE_INTEREST_RATE = {g: round(sum(v.values()) / len(v), 2) for g, v in GRADE_INTENT_INTEREST_RATE.items()}


def estimated_interest_rate(loan_grade: str, loan_intent: str = None) -> float:
    """Auto-computed interest rate for a given loan grade + loan intent,
    based on the real grade/intent averages above. Falls back to the
    grade-only average if intent is omitted or unrecognized."""
    by_intent = GRADE_INTENT_INTEREST_RATE.get(loan_grade)
    if by_intent and loan_intent in by_intent:
        return by_intent[loan_intent]
    return GRADE_INTEREST_RATE.get(loan_grade, 12.0)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's Section 5-6 cleaning steps on the raw
    cr_loan.csv. Used only by train_model.py."""
    clean = df.copy()

    # Remove duplicate records
    clean = clean.drop_duplicates()

    # Remove implausible outliers (clear data-entry errors)
    clean = clean[clean["person_age"] <= 100]
    clean = clean[(clean["person_emp_length"] <= 60) | (clean["person_emp_length"].isna())]

    # Impute missing values with the median (robust to skew/outliers)
    clean["person_emp_length"] = clean["person_emp_length"].fillna(clean["person_emp_length"].median())
    clean["loan_int_rate"] = clean["loan_int_rate"].fillna(clean["loan_int_rate"].median())

    return clean.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the ordinal / binary / one-hot encoding used at training time.
    Safe to call on either a full training dataframe or a single-row
    inference dataframe — one-hot columns are created the same way in
    both cases; missing columns get reconciled by align_columns()."""
    encoded = df.copy()

    encoded["loan_grade"] = encoded["loan_grade"].map(GRADE_ORDER)
    encoded["cb_person_default_on_file"] = encoded["cb_person_default_on_file"].map(DEFAULT_FLAG_MAP)

    encoded = pd.get_dummies(encoded, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    onehot_cols = [c for c in encoded.columns
                   if c.startswith("person_home_ownership_") or c.startswith("loan_intent_")]
    if onehot_cols:
        encoded[onehot_cols] = encoded[onehot_cols].astype(int)

    return encoded


def align_columns(df: pd.DataFrame, trained_columns: list) -> pd.DataFrame:
    """Reindex a single applicant's encoded row to match the exact column
    order/set the model was trained on. Any one-hot column not present for
    this applicant (e.g. a home-ownership category not selected) is filled
    with 0, matching how the model saw it during training."""
    return df.reindex(columns=trained_columns, fill_value=0).astype(float)


def build_applicant_row(
    person_age: int,
    person_income: float,
    person_home_ownership: str,
    person_emp_length: float,
    loan_intent: str,
    loan_grade: str,
    loan_amnt: float,
    loan_int_rate: float,
    loan_percent_income: float,
    cb_person_default_on_file: str,
    cb_person_cred_hist_length: int,
) -> pd.DataFrame:
    """Build a single-row raw-feature DataFrame from the Prediction page's
    widget values, in the same column layout the training data used."""
    return pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }])


def risk_level(probability: float) -> str:
    """Collapse the notebook's 4-tier segmentation (Low/Medium/High/Critical)
    into the 3 tiers requested for this app: High absorbs both High and
    Critical, since the demo only needs Low / Medium / High."""
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    else:
        return "High"
