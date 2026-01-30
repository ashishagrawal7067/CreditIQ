"""
CreditIQ — Data Preprocessing & Feature Engineering Module
Reusable functions for consumer credit risk modeling on Lending Club datasets.
"""

import pandas as pd
import numpy as np


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw Lending Club data to completed loans and compute base targets.
    
    Eliminates ongoing loans ('Current', 'Late') to prevent lookahead target leakage.
    """
    known_outcomes = ['Fully Paid', 'Charged Off']
    df_clean = df[df['loan_status'].isin(known_outcomes)].copy()
    
    # Binary target: 1 = default / charged off, 0 = fully paid
    df_clean['default'] = (df_clean['loan_status'] == 'Charged Off').astype(int)
    
    # Date parsing
    df_clean['issue_d'] = pd.to_datetime(df_clean['issue_d'], format='%b-%Y')
    df_clean['issue_year'] = df_clean['issue_d'].dt.year
    
    # FICO midpoint
    if 'fico_range_low' in df_clean.columns and 'fico_range_high' in df_clean.columns:
        df_clean['fico_mid'] = (df_clean['fico_range_low'] + df_clean['fico_range_high']) / 2.0
        
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 8 domain-specific credit risk and borrower stability features.
    
    Caps extreme ratio outliers at the 99th percentile to ensure stability.
    """
    df_feat = df.copy()
    
    # 1. Credit History Length (months)
    if 'earliest_cr_line' in df_feat.columns:
        df_feat['earliest_cr_line'] = pd.to_datetime(df_feat['earliest_cr_line'], format='%b-%Y')
        df_feat['credit_age_months'] = (
            (df_feat['issue_d'].dt.year - df_feat['earliest_cr_line'].dt.year) * 12 +
            (df_feat['issue_d'].dt.month - df_feat['earliest_cr_line'].dt.month)
        )
        df_feat['credit_age_months'] = df_feat['credit_age_months'].clip(lower=0)

    # 2. Borrower Capacity & Cash Flow Burden
    df_feat['loan_to_income'] = df_feat['loan_amnt'] / df_feat['annual_inc'].replace(0, np.nan)
    df_feat['monthly_inc'] = df_feat['annual_inc'] / 12.0
    df_feat['instalment_to_income'] = df_feat['installment'] / df_feat['monthly_inc'].replace(0, np.nan)
    
    # Outlier capping (99th percentiles)
    df_feat['loan_to_income'] = df_feat['loan_to_income'].clip(upper=0.50)
    df_feat['instalment_to_income'] = df_feat['instalment_to_income'].clip(upper=0.1994)

    # 3. Delinquency and Public Record Flags
    df_feat['delinq_flag'] = (df_feat['delinq_2yrs'] > 0).astype(int)
    df_feat['pub_rec_flag'] = (df_feat['pub_rec'] > 0).astype(int)

    # 4. Loan Structure
    if df_feat['term'].dtype == object:
        df_feat['term_months'] = df_feat['term'].str.extract(r'(\d+)').astype(int)
    else:
        df_feat['term_months'] = df_feat['term']

    grade_avg_rates = {
        'A': 7.49, 'B': 11.72, 'C': 15.25,
        'D': 19.47, 'E': 23.38, 'F': 27.13, 'G': 29.08
    }
    df_feat['int_rate_grade_dev'] = df_feat['int_rate'] - df_feat['grade'].map(grade_avg_rates).fillna(13.0)

    return df_feat


def compute_amortized_installment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    """Compute exact monthly amortization payment for a fixed-rate loan."""
    monthly_r = (annual_rate_pct / 100.0) / 12.0
    if monthly_r > 0:
        return principal * (monthly_r * ((1.0 + monthly_r) ** term_months)) / (((1.0 + monthly_r) ** term_months) - 1.0)
    return principal / term_months

