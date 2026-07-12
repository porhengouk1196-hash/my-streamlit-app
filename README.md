# Credit Risk Prediction — Streamlit Banking Demo

A single-page Streamlit app for predicting loan default risk, matching the
analysis and results from the accompanying Jupyter notebook
(`Credit_Risk_Prediction09`).

## Project Structure

```
credit_risk_app/
├── app.py                   # The entire app — form, model selection, results
├── requirements.txt
├── README.md
├── cr_loan.csv               # <- dataset (already included)
├── assets/
│   └── style.css             # Navy / white / light grey / sky-blue theme
├── model/
│   ├── train_model.py        # Reproduces the notebook's pipeline, trains all 4 models
│   └── credit_risk_model.pkl # Created by train_model.py (not included — see Setup)
└── utils/
    ├── preprocessing.py      # Shared cleaning/encoding — used by training AND inference
    ├── model_loader.py       # Cached model bundle loader
    └── styling.py             # Card/theme HTML helpers
```

## Why a training script is included instead of a pre-trained `.pkl`

This project was assembled in an environment without internet access or
scikit-learn installed, so a real trained model file could not be produced
and shipped directly. `model/train_model.py` reproduces the exact cleaning
and encoding pipeline documented in the notebook — running it locally
regenerates all 4 models with results matching what's documented in the
notebook (XGBoost around 93.8% accuracy / 95.3% ROC-AUC).

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train the models (also works if you already have `cr_loan.csv` — it's
   included in this project already):

   ```bash
   python model/train_model.py
   ```

   This prints a model comparison table and saves all 4 trained models to
   `model/credit_risk_model.pkl`.

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## What the app does

- Enter applicant details (age, income, home ownership, employment length,
  loan intent, loan grade, loan amount, credit history length, previous
  default). Hover the "?" next to any field for a short explanation of what
  it means.
- **Interest rate is set automatically from Loan Grade + Loan Intent** —
  not a free input. It uses each grade/intent combination's real average
  rate computed directly from `cr_loan.csv`. Grade is still the dominant
  driver (13+ points end to end), and intent shifts the rate by a smaller,
  real amount within each grade — so changing Loan Intent visibly moves
  the auto-filled rate.
- Choose which trained model scores the application (Logistic Regression,
  Decision Tree, Random Forest, or XGBoost) — each option shows its
  accuracy and ROC-AUC so you know what you're picking.
- Click **Predict** to see: Default/Non-Default, default probability, a
  confidence score, and a Low/Medium/High risk level, shown as cards.
- A plain-language explanation of what that risk level means, followed by
  **Recommended Actions** — concrete, computed guidance such as the maximum
  loan amount or minimum income that would bring this specific applicant to
  Low risk (found by re-scoring the model, not a fixed rule), plus flags
  for a high loan-to-income ratio or a prior default on file.
- A **Score by Model** chart showing this same applicant's default
  probability under all 4 models at once, so you can see how much the
  result changes depending on which model is used — with an expander below
  it for the exact numbers.

## Design Notes

- **Palette:** matches the project slide deck — Navy Blue (`#0B1F3A`), White
  (`#FFFFFF`), Light Grey (`#F5F6F8`), and a single Sky Blue accent
  (`#2E86DE`) — see `assets/style.css`. Low/Medium results stay within
  this fixed palette, differentiated by visual weight. **High risk is the
  one deliberate exception:** it renders in red (`#C0392B`) as a clear
  alert, since flagging genuinely risky applicants outweighs strict
  palette purity here.
- **No animations**, minimal icons, single page — no sidebar navigation,
  since the app does one thing only: score an applicant.
- **Train/serve consistency:** `utils/preprocessing.py` is imported by both
  `model/train_model.py` and `app.py`, so the exact same cleaning/encoding
  logic runs at training time and at prediction time.

## Notes / Limitations

- The Loan Percent Income field is calculated automatically from Income and
  Loan Amount rather than entered manually, so the value the model sees can
  never contradict the other two fields.
- This is a demonstration application. It is not connected to any real
  banking system and should not be used for actual lending decisions.
