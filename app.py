"""
Credit Risk Prediction — Banking Demo Application
====================================================
A single-page prediction tool: enter applicant details, pick which trained
model to score with, and see the result, a plain-language explanation,
concrete recommendations, and a comparison chart across all 4 models.

Run with:
    streamlit run app.py
"""

import os
import sys

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from utils.model_loader import load_model_bundle  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    HOME_OWNERSHIP_OPTIONS,
    LOAN_GRADE_OPTIONS,
    LOAN_INTENT_OPTIONS,
    align_columns,
    build_applicant_row,
    encode_features,
    estimated_interest_rate,
    risk_level,
)
from utils.styling import load_css, render_card, render_result_card, section_divider  # noqa: E402

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon=None,  # minimal icons per spec
    layout="wide",
)

CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
load_css(CSS_PATH)

NAVY = "#0B1F3A"
ACCENT = "#2E86DE"
GREY = "#F5F6F8"

# What each input represents — shown as a hover tooltip next to the field,
# so the form stays compact instead of a wall of explanatory text.
FEATURE_HELP = {
    "age": "Applicant's age in years.",
    "income": "Applicant's annual income, in dollars.",
    "home_ownership": "Whether the applicant rents, owns, or has a mortgage on their home.",
    "emp_length": "Number of years in the applicant's current job.",
    "loan_intent": "The reason the applicant is borrowing (e.g. education, medical, venture).",
    "loan_grade": "Lender-assigned credit grade for this loan, A (best) to G (worst).",
    "loan_amount": "The amount of money being requested.",
    "interest_rate": "Set automatically from Loan Grade and Loan Intent, using each combination's "
                      "real average rate in the training data (grade drives most of the difference; "
                      "intent shifts it by a smaller amount).",
    "loan_percent_income": "Loan amount as a share of annual income — calculated automatically "
                            "from Income and Loan Amount above.",
    "credit_history_length": "Number of years of recorded credit history.",
    "previous_default": "Whether the applicant has defaulted on a loan before.",
}

# Plain-language explanation of what each risk tier means and what action
# it typically implies, shown after a prediction is made.
RESULT_MEANING = {
    "Low": "The model sees this applicant as low risk. Standard approval terms are appropriate.",
    "Medium": "The model sees some risk here. Consider approving with closer monitoring or "
              "adjusted terms rather than an automatic decline.",
    "High": "The model sees this applicant as high risk of default. Manual review is recommended "
            "before any approval decision.",
}

TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}


def _predict(bundle: dict, model_name: str, raw_row: pd.DataFrame) -> float:
    """Encode one applicant row exactly as at training time and return the
    chosen model's predicted probability of default."""
    encoded = encode_features(raw_row)
    encoded = encoded.drop(columns=["loan_status"], errors="ignore")
    aligned = align_columns(encoded, bundle["columns"])

    model = bundle["models"][model_name]
    if model_name == "Logistic Regression" and bundle.get("scaler") is not None:
        aligned = bundle["scaler"].transform(aligned)

    probability = model.predict_proba(aligned)[:, 1][0]
    return float(probability)


def _row_with_overrides(base_kwargs: dict, **overrides) -> pd.DataFrame:
    kwargs = dict(base_kwargs)
    kwargs.update(overrides)
    return build_applicant_row(**kwargs)


def _max_loan_for_low_risk(bundle, model_name, base_kwargs, current_amount) -> Optional[int]:
    """Binary search the largest loan amount (at or below the current one)
    that would keep this applicant at Low risk, holding everything else
    fixed. Returns None if even the smallest loan amount doesn't reach Low."""
    lo, hi = 500, int(current_amount)
    row_lo = _row_with_overrides(base_kwargs, loan_amnt=lo,
                                  loan_percent_income=lo / base_kwargs["person_income"]
                                  if base_kwargs["person_income"] > 0 else 0.0)
    if risk_level(_predict(bundle, model_name, row_lo)) != "Low":
        return None
    for _ in range(20):
        mid = (lo + hi) // 2
        if mid == lo:
            break
        pct = mid / base_kwargs["person_income"] if base_kwargs["person_income"] > 0 else 0.0
        row = _row_with_overrides(base_kwargs, loan_amnt=mid, loan_percent_income=pct)
        if risk_level(_predict(bundle, model_name, row)) == "Low":
            lo = mid
        else:
            hi = mid
    return lo


def _min_income_for_low_risk(bundle, model_name, base_kwargs, current_income) -> Optional[int]:
    """Binary search the smallest annual income (at or above the current
    one) that would bring this applicant to Low risk at their current loan
    amount. Returns None if even a very high income doesn't reach Low."""
    lo, hi = int(current_income), 1_000_000
    pct_hi = base_kwargs["loan_amnt"] / hi if hi > 0 else 0.0
    row_hi = _row_with_overrides(base_kwargs, person_income=hi, loan_percent_income=pct_hi)
    if risk_level(_predict(bundle, model_name, row_hi)) != "Low":
        return None
    for _ in range(20):
        mid = (lo + hi) // 2
        if mid == hi:
            break
        pct = base_kwargs["loan_amnt"] / mid if mid > 0 else 0.0
        row = _row_with_overrides(base_kwargs, person_income=mid, loan_percent_income=pct)
        if risk_level(_predict(bundle, model_name, row)) == "Low":
            hi = mid
        else:
            lo = mid
    return hi


def _comparison_chart(rows: list[dict]):
    names = [r["Model"] for r in rows]
    probs = [r["_prob"] for r in rows]
    colors = [NAVY if p >= 0.5 else ACCENT for p in probs]

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(names, probs, color=colors)
    ax.axvline(0.5, color="#B42A2A", linewidth=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Default Probability")
    ax.set_facecolor(GREY)
    fig.patch.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for i, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.93), i, f"{p:.0%}", va="center", fontsize=9, color=NAVY)
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Credit Risk Prediction")
    st.markdown(
        '<p class="muted">Enter applicant details, choose a model, and click Predict.</p>',
        unsafe_allow_html=True,
    )
    section_divider()

    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            "No trained model found. Run `python model/train_model.py` from the project root "
            "first — see README.md for setup steps."
        )
        return

    model_names = list(bundle["models"].keys())
    metrics = bundle.get("metrics", {})

    with st.form("prediction_form"):
        st.subheader("Model")
        model_name = st.selectbox(
            "Choose which trained model scores this application",
            model_names,
            index=model_names.index(bundle["best_model_name"]) if bundle["best_model_name"] in model_names else 0,
            help="Switch models and predict again to see how the score changes.",
        )
        m = metrics.get(model_name, {})
        if m:
            st.caption(f"{model_name} — Accuracy {m.get('Accuracy', 0):.1%}, ROC AUC {m.get('ROC AUC', 0):.1%}")

        st.subheader("Borrower Profile")
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.slider("Age", min_value=18, max_value=100, value=30, step=1,
                                    help=FEATURE_HELP["age"])
            person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000,
                                             help=FEATURE_HELP["income"])
        with col2:
            person_home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP_OPTIONS,
                                                   help=FEATURE_HELP["home_ownership"])
            person_emp_length = st.slider("Employment Length (years)", min_value=0.0, max_value=40.0,
                                           value=5.0, step=0.5, help=FEATURE_HELP["emp_length"])

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.subheader("Loan Details")
        col3, col4 = st.columns(2)
        with col3:
            loan_intent = st.selectbox("Loan Intent", LOAN_INTENT_OPTIONS, help=FEATURE_HELP["loan_intent"])
            loan_grade = st.selectbox("Loan Grade", LOAN_GRADE_OPTIONS, help=FEATURE_HELP["loan_grade"])
            cb_person_default_on_file = st.selectbox("Previous Default", ["No", "Yes"],
                                                       help=FEATURE_HELP["previous_default"])
        with col4:
            loan_amnt = st.slider("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500,
                                   help=FEATURE_HELP["loan_amount"])
            loan_int_rate = estimated_interest_rate(loan_grade, loan_intent)
            st.metric("Interest Rate (auto, by grade + intent)", f"{loan_int_rate:.2f}%", help=FEATURE_HELP["interest_rate"])
            cb_person_cred_hist_length = st.slider("Credit History Length (years)", min_value=0,
                                                     max_value=40, value=5, step=1,
                                                     help=FEATURE_HELP["credit_history_length"])

        loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0.0
        st.metric("Loan Percent Income (derived)", f"{loan_percent_income:.2%}", help=FEATURE_HELP["loan_percent_income"])

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    base_kwargs = dict(
        person_age=person_age,
        person_income=person_income,
        person_home_ownership=person_home_ownership,
        person_emp_length=person_emp_length,
        loan_intent=loan_intent,
        loan_grade=loan_grade,
        loan_amnt=loan_amnt,
        loan_int_rate=loan_int_rate,
        loan_percent_income=loan_percent_income,
        cb_person_default_on_file="Y" if cb_person_default_on_file == "Yes" else "N",
        cb_person_cred_hist_length=cb_person_cred_hist_length,
    )
    raw_row = build_applicant_row(**base_kwargs)

    try:
        probability = _predict(bundle, model_name, raw_row)
    except Exception as exc:  # noqa: BLE001 - show the error instead of a raw traceback page
        st.error(f"Prediction failed: {exc}")
        return

    predicted_label = "Default" if probability >= 0.5 else "Non-Default"
    confidence = max(probability, 1 - probability)
    tier = risk_level(probability)

    section_divider()
    st.subheader("Result")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_result_card("Prediction", predicted_label, tone="high" if predicted_label == "Default" else "low")
    with c2:
        render_result_card("Default Probability", f"{probability:.1%}", tone="high" if probability >= 0.5 else "low")
    with c3:
        render_result_card("Confidence Score", f"{confidence:.1%}", tone="medium")
    with c4:
        tone = {"Low": "low", "Medium": "medium", "High": "high"}[tier]
        render_result_card("Risk Level", tier, tone=tone)

    st.caption(f"Scored using: {model_name}")

    render_card("What this means", RESULT_MEANING[tier])

    # ---- Recommended actions: concrete, computed from this applicant's own numbers ----
    st.subheader("Recommended Actions")
    action_lines = []

    if loan_percent_income > 0.35:
        max_amnt_35 = int(person_income * 0.35)
        action_lines.append(
            f"Loan-to-income ratio is {loan_percent_income:.0%}, above the 35% level associated with "
            f"materially higher default rates in this data. Reducing the loan amount to about "
            f"${max_amnt_35:,} would bring it under 35%."
        )

    if tier != "Low":
        max_loan = _max_loan_for_low_risk(bundle, model_name, base_kwargs, loan_amnt)
        if max_loan is not None and max_loan < loan_amnt:
            action_lines.append(f"At this income, reducing the loan amount to about ${max_loan:,} or "
                                 f"below would bring this applicant to Low risk.")
        min_income = _min_income_for_low_risk(bundle, model_name, base_kwargs, person_income)
        if min_income is not None and min_income > person_income:
            action_lines.append(f"At this loan amount, an annual income of about ${min_income:,} or "
                                 f"higher would bring this applicant to Low risk.")
        if max_loan is None and min_income is None:
            action_lines.append("Adjusting loan amount or income alone does not bring this applicant to "
                                 "Low risk — other factors (grade, prior default, credit history) are driving "
                                 "the score.")

    if cb_person_default_on_file == "Yes":
        action_lines.append("This applicant has a prior default on file. In this data, a prior default "
                             "roughly doubles the historical default rate (≈38% vs ≈18%) — recommend "
                             "additional documentation regardless of score.")

    if tier == "Low":
        action_lines.append("No changes needed based on loan amount or income — this profile already "
                             "scores Low risk. Standard approval terms apply.")

    for line in action_lines:
        render_card("→", line)

    # ---- Chart: how the score changes across all 4 models ----
    st.subheader("Score by Model")
    comparison_rows = []
    for name in model_names:
        try:
            p = _predict(bundle, name, raw_row)
            comparison_rows.append({"Model": name, "_prob": p})
        except Exception:  # noqa: BLE001 - skip a model that fails rather than break the chart
            continue
    if comparison_rows:
        st.pyplot(_comparison_chart(comparison_rows), use_container_width=True)

    with st.expander("Exact numbers per model"):
        table_rows = [{"Model": r["Model"], "Default Probability": f"{r['_prob']:.1%}",
                        "Risk Level": risk_level(r["_prob"])} for r in comparison_rows]
        st.table(pd.DataFrame(table_rows).set_index("Model"))


if __name__ == "__main__":
    main()
