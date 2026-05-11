# health-insurance-cost-prediction

## 📋 Project Overview

Shield Insurance's underwriters were calculating premiums manually — slow, inconsistent, and expensive. This project builds an ML-powered system that predicts a customer's annual health insurance premium instantly from their profile, deployed as an interactive Streamlit web application.

**Example:**
> Given Rahul, Age 26 | Male | Non-smoker | Diabetic | Bronze Plan | ₹6L income | Northwest region  
> → **Predicted Annual Premium: ₹9,053**


## 🏗️ Project Architecture

The system uses **two separate models** routed by customer age:

```
Customer Input
      │
      ▼
  Age ≤ 25?
  ├── YES → model_young.joblib  (XGBoost + Genetical Risk feature)
  └── NO  → model_rest.joblib   (XGBoost)
      │
      ▼
Predicted Premium (₹)
```

---

## 🔁 ML Pipeline

### 1. Data Cleaning
- Drop missing values and duplicates
- Fix negative dependant counts (`.abs()`)
- Remove age outliers (> 100)
- Remove income outliers (99.9th percentile — a business decision over IQR)
- Harmonise smoking status spellings (`Not Smoking`, `Does Not Smoke`, `Smoking=0` → `No Smoking`)

### 2. Feature Engineering

**Medical Risk Score** — converts raw medical history text into a numerical risk score:

| Condition | Risk Score |
|---|---|
| Heart Disease | 8 |
| Diabetes | 6 |
| High Blood Pressure | 6 |
| Thyroid | 5 |
| No Disease | 0 |

Scores are normalised to 0–1 range.

**Encoding:**
- Ordinal: `insurance_plan` (Bronze=1, Silver=2, Gold=3), `income_level` (1–4)
- One-Hot: `gender`, `region`, `marital_status`, `bmi_category`, `smoking_status`, `employment_status`

**Scaling:** MinMaxScaler on `age`, `number_of_dependants`, `income_level`, `income_lakhs`, `insurance_plan`

**VIF Check:** `income_level` dropped due to high multicollinearity with `income_lakhs`

### 3. Model Training

| Group | Model | Extreme Error Rate |
|---|---|---|
| All data (baseline) | Linear Regression | 73% on young customers ❌ |
| Age ≤ 25 (without Genetical Risk) | XGBoost | 73% ❌ |
| Age ≤ 25 (with Genetical Risk) | XGBoost + RandomizedSearchCV | **2%** ✅ |
| Age > 25 | XGBoost + RandomizedSearchCV | **0.3%** ✅ |

> **Extreme error** = prediction more than 10% off from actual premium

---

## 🚨 The Key Discovery — Model Segmentation

Training a single model on all 50,000 records produced R² = 0.97 — which looked excellent. But residual analysis revealed:

- **Age ≤ 25 (20,096 customers):** 73% extreme error rate
- **Age > 25 (29,904 customers):** 0.3% extreme error rate

The majority group's accuracy was masking a catastrophic failure on young customers. Deploying without this analysis would have meant Shield Insurance **losing crores** on mispriced policies.

**Solution:** Segment the data, train two separate models, and request an additional feature (`Genetical_Risk`) from the business for the young group — reducing errors from 73% → 2%.

---
---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib openpyxl statsmodels
```

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser. Enter a customer's details and click **Calculate Premium** to get an instant prediction.

> **Note:** The `Genetical Risk` field only appears for customers aged 25 and under, as it is only used by the young group model.

---

## 📐 Metrics Reference

| Metric | Description |
|---|---|
| **R² Score** | Proportion of variance explained. Above 0.90 is generally good. |
| **Extreme Error %** | % of predictions > 10% off actual. Below 5% is acceptable. |
| **diff_pct** | `(predicted - actual) / actual × 100`. Positive = over-predicted. |
| **VIF** | Variance Inflation Factor. > 5 investigate; > 10 drop. |

---

## 💡 Key Lessons

- **Overall accuracy can lie.** Always compute per-segment metrics — a 97% R² hid a 73% failure rate.
- **Residual analysis is mandatory.** Quantify extreme errors, not just headline scores.
- **Feature discovery beats model tuning.** One new column (Genetical Risk) reduced errors from 73% → 2%. No hyperparameter search could have achieved that.
- **Save scalers with column names.** Re-fitting a scaler at prediction time produces wrong results.
- **Knowing when not to deploy is a core skill.** The rest group model could have gone live immediately while the young model was being fixed.

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, statsmodels
- **Streamlit** — interactive web app
- **joblib** — model and scaler serialisation
---

live demo link - https://health-insurance-cost-prediction-ywgdbcant8n7ebmb2pz692.streamlit.app/
