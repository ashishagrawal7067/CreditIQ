# CreditIQ — Lending Club Credit Risk Analysis

[![GitHub Repository](https://img.shields.io/badge/GitHub-ashishagrawal7067%2FCreditIQ-181717?style=flat&logo=github)](https://github.com/ashishagrawal7067/CreditIQ)
[![Live Demo](https://img.shields.io/badge/Streamlit-CreditIQ%20App-FF4B4B?style=flat&logo=streamlit)](https://creditiq-risk-analyser.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Can we predict loan default before it happens — and quantify the business value of doing so?**

This project builds a production-style credit default prediction system on ~1.3 million 
real loan records from Lending Club (2007–2018). It covers the full data science lifecycle: 
exploratory analysis, feature engineering, machine learning modelling, model interpretability, 
hyperparameter tuning, temporal validation, and a financial simulation that translates model 
output into lending P&L impact.

---

## Live Demo

🚀 **[Try the CreditIQ Risk Analyser](https://creditiq-risk-analyser.streamlit.app)** *(Deployed via Streamlit Community Cloud from [ashishagrawal7067/CreditIQ](https://github.com/ashishagrawal7067/CreditIQ))*

Enter borrower details and get an instant default probability prediction powered by XGBoost 
trained on 1.3M+ real loans. Features include:
- Real-time default probability gauge with risk tier classification
- Risk signal dashboard across 6 key borrower metrics
- Lender decision simulator with adjustable approval threshold
- Portfolio context charts comparing this loan to historical grade averages
- Full model performance summary

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
**Records:** ~1.3 million loans with known outcomes (Fully Paid or Charged Off)  
**Period:** 2007 – 2018  
**Features:** 141 raw features covering borrower profile, credit history, loan terms, and repayment behaviour  

> Raw data files are not committed to this repository. Download from the link above 
> and place in `data/raw/`.

---

## Project Structure
CreditIQ/
* notebooks/
    * 01_eda.ipynb *(Exploratory analysis & storytelling)*
   * 02_feature_engineering.ipynb *(Feature creation & selection)*
   * 03_Modelling.ipynb *(Model training & comparison)*
   * 04_model_interpretation.ipynb *(SHAP, scorecard, business simulation)*
   * 05_tuning.ipynb *(Optuna hyperparameter tuning)*
   * 06_temporal_validation.ipynb *(Model stability across time)*
* src/
   * preprocessing.py  *(Reusable cleaning functions)*
   * model_utils.py *(Evaluation & plotting helpers)*
* sql/ *(MySQL scripts for data analysis)*
* app.py *(Streamlit web application)*
* reports/
    * figures/ *(All saved charts) (17 figures)*
* data/
   * README.md *(Data source & download instructions)*
* requirements.txt
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

### Temporal Validation
Model trained on 2007–2015 loans, tested on each subsequent year independently:

| Test Year | Loans | Default Rate | Gini | vs Random Split |
|-----------|-------|-------------|------|-----------------|
| 2016 | 293,095 | 23.28% | 0.4234 | -0.0192 |
| 2017 | 169,300 | 23.12% | 0.4016 | -0.0410 |
| 2018 | 56,311 | 15.75% | 0.3972 | -0.0454 |

Concept drift detected from 2017 onward. **Recommendation: quarterly retraining 
on a rolling 36-month window**, with a Gini alert threshold of -3pp on a 90-day holdout.

### What Drives Default Risk
Based on SHAP value analysis, the five most influential signals at origination:

1. **Interest rate** — higher rate loans carry significantly more default risk
2. **Loan term** — 60-month loans default at nearly twice the rate of 36-month
3. **Loan grade** — Lending Club's own grading carries strong independent signal
4. **FICO score** — lower scores at origination independently predict default
5. **Loan-to-income ratio** — affordability is a primary driver of repayment

### Business Simulation
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

## Methodology

| Phase | Description |
|-------|-------------|
| **EDA** | Default rate analysis across grade, time, FICO, purpose, and geography |
| **Feature Engineering** | Credit age, loan-to-income ratio, utilisation segments, delinquency flags |
| **Modelling** | Logistic Regression → Random Forest → XGBoost with scale_pos_weight |
| **Tuning** | Optuna 50-trial Bayesian search — baseline confirmed optimal at scale |
| **Temporal Validation** | Trained 2007–2015, tested 2016/2017/2018 independently |
| **Interpretation** | SHAP values, feature importance, credit scorecard |
| **Business Simulation** | P&L comparison: approve-all baseline vs. model-gated strategy |
| **Deployment** | Live Streamlit app with real-time loan scoring |

---

## Limitations & Future Work

### Current Limitations

**Survivorship bias / Reject Inference**
This model trains exclusively on approved loans. We have no outcome data for declined 
applicants. Techniques such as augmentation or parcelling would be required to correct 
for this bias in a production setting.

**Loss Given Default assumption**
The business simulation assumes 100% loss on defaulted loans. In practice, recovery 
rates average 30–40 cents on the dollar. A full expected loss model would incorporate:
`Expected Loss = PD × EAD × LGD`

**Concept drift**
Temporal validation revealed Gini degradation from 0.4426 to 0.3972 between the 
training window and 2018 loans. Quarterly retraining recommended.

### Future Work
- **Survival Analysis** — model time-to-default using Cox Proportional Hazards regression
- **Full Expected Loss Framework** — separate PD, LGD, and EAD models (Basel III standard)
- **Interactive Power BI Dashboard** — executive-facing dashboard with live filtering
- **Model Monitoring Pipeline** — automated Gini tracking with retraining alerts

---

## Setup

```bash
git clone https://github.com/ashishagrawal7067/CreditIQ.git
cd CreditIQ
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place in `data/`, then:

```bash
# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb

# Or launch the Streamlit app
streamlit run app.py
```

---

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Optuna · Matplotlib · Seaborn · Streamlit · MySQL