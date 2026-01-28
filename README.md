# Lending Club Credit Risk — End-to-End Analysis

> **Can we predict loan default before it happens? Can the business value of doing so be quantified?**

This project builds a production-style credit default prediction system on ~2.26 million 
real loan records from Lending Club (2007–2018). It covers the full data science lifecycle: 
exploratory analysis, feature engineering, machine learning modelling, model interpretability, 
and a financial simulation that translates model output into lending P&L impact.

---

## Business Context

Consumer lending is a risk pricing problem. A lender that cannot distinguish a likely 
defaulter from a likely repayer will either turn away good borrowers (too conservative) 
or absorb losses on bad ones (too permissive). This project addresses that problem directly:

- **What is the probability this borrower defaults?**
- **Which borrower characteristics drive that risk?**
- **What is the financial impact of acting on model predictions vs. approving all loans?**

---

## Dataset

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**Records:** ~2.26 million loans with known outcomes (Fully Paid or Charged Off)  
**Period:** 2007 – 2018  
**Features:** 141 raw features covering borrower profile, credit history, loan terms, and repayment behaviour  

> Raw data files are not committed to this repository. Download [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download) 
> and place in `data/`.

---

## Project Structure
**lending-club-credit-risk/**
- notebooks/
    - 01_eda.ipynb (*Exploratory analysis & storytelling*)
    - 02_feature_engineering.ipynb (*Feature creation & selection*)
    - 03_modelling.ipynb (*Model training & comparison*)
    - 04_model_interpretation.ipynb (*SHAP, scorecard, business simulation*)
- src/
    -  preprocessing.py (*Reusable cleaning function*)
    - model_utils.py (*Evaluation & plotting helpers*)
- reports/
    - figures/ (*All saved charts*)
- data/
    - README.md (*Data source & download instructions*)
- requirements.txt

---

## Methodology

| Phase | Description |
|-------|-------------|
| **EDA** | Default rate analysis across grade, time, FICO, purpose, and geography |
| **Feature Engineering** | Credit age, loan-to-income ratio, utilisation segments, delinquency flags |
| **Modelling** | Logistic Regression → Random Forest → XGBoost with Optuna tuning |
| **Evaluation** | AUC-ROC, Gini coefficient, Precision-Recall curve |
| **Interpretation** | SHAP values, feature importance, credit scorecard |
| **Business Simulation** | P&L comparison: approve-all baseline vs. model-gated strategy |

---

## Key Results

### Model Performance
| Model | AUC-ROC | Gini Coefficient | PR-AUC |
|-------|---------|-----------------|--------|
| Logistic Regression (baseline) | 0.7081 | 0.4163 | 0.3692 |
| Random Forest | 0.7115 | 0.4230 | 0.3758 |
| **XGBoost (selected)** | **0.7213** | **0.4426** | **0.3903** |
| XGBoost (Optuna tuned) | 0.7201 | 0.4401 | 0.3880 |

> Gini Coefficient is the primary evaluation metric — the industry standard
> in consumer credit risk. A Gini of 0 = random; 1.0 = perfect.
> Industry models typically achieve 0.40–0.65.
> Optuna tuning on a 200k subsample did not improve on the baseline —
> confirming default parameters were already well-calibrated at full scale.

### Temporal Validation
Model trained on 2007–2015 loans, tested on each subsequent year independently:

| Test Year | Loans | Default Rate | Gini | vs Random Split |
|-----------|-------|-------------|------|-----------------|
| 2016 | 293,095 | 23.28% | 0.4234 | -0.0192 |
| 2017 | 169,300 | 23.12% | 0.4016 | -0.0410 |
| 2018 | 56,311 | 15.75% | 0.3972 | -0.0454 |

Concept drift detected from 2017 onward — consistent with the 2016–2018
credit cycle shift. **Recommendation: quarterly retraining on a rolling
36-month window**, with a Gini alert threshold of -3pp on a 90-day holdout.

### What Drives Default Risk
Based on SHAP value analysis, the five most influential signals at origination:

1. **Interest rate** — higher rate loans carry significantly more default risk
2. **Loan term** — 60-month loans default at nearly twice the rate of 36-month
3. **Loan grade** — Lending Club's own grading carries strong independent signal
4. **FICO score** — lower scores at origination independently predict default
5. **Loan-to-income ratio** — affordability is a primary driver of repayment

### Business Simulation
Simulating two lending strategies on a 269,062-loan held-out test set:

| Strategy | Return Rate | On a $1B Portfolio |
|----------|------------|-------------------|
| Approve All (baseline) | 16.27% | $162.7M net return |
| **Model-Gated (threshold = 35%)** | **18.06%** | **$180.6M net return** |

**The model generates $17.9M in additional value per $1B deployed** — by
declining the highest-risk 68.5% of applications while reducing the realised
default rate from 19.96% to 7.14%.

> Assumptions: interest revenue = rate × principal × term;
> LGD = 100% (no recovery); no cost of capital modelled.
---

## Setup

```bash
git clone https://github.com/Dunmxie/lending_club_credit_risk.git
cd lending_club_credit_risk
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download) and place the `.csv.gz` file in `data/`.

---

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Optuna