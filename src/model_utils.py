"""
CreditIQ — Model Evaluation & Governance Utilities
Functions for credit risk benchmarking, regulatory scorecard conversion, and portfolio simulation.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def evaluate_credit_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Train, predict, and compute standard consumer credit risk metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    gini = 2.0 * auc - 1.0
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    return {
        "name": name,
        "model": model,
        "y_prob": y_prob,
        "auc": auc,
        "gini": gini,
        "pr_auc": pr_auc,
        "brier": brier,
        "elapsed_seconds": elapsed,
    }


def compute_credit_scorecard(
    log_odds: np.ndarray,
    base_score: float = 600.0,
    pdo: float = 20.0,
    base_odds: float = 4.0
) -> np.ndarray:
    """Transform log-odds of default into a points-based credit score.
    
    Standard Banking Methodology:
        Odds = Good:Bad = P(paid) / P(default).
        Since log_odds from binary logistic regression equals ln(P(default)/P(paid)),
        ln(Good/Bad) = -log_odds.
        
        Score = Offset + Factor * ln(Good/Bad) = Offset - Factor * log_odds.
        
    Higher score indicates higher repayment probability (lower default risk).
    """
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset - factor * log_odds


def simulate_portfolio_strategy(
    df_test: pd.DataFrame,
    pred_prob_col: str = 'pred_prob',
    actual_default_col: str = 'default',
    loan_amnt_col: str = 'loan_amnt',
    int_rate_col: str = 'int_rate',
    term_months_col: str = 'term_months',
    thresholds: list = None,
    portfolio_size: float = 1_000_000_000.0
) -> pd.DataFrame:
    """Simulate P&L across approval thresholds and scale to an illustrative portfolio."""
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    df_sim = df_test.copy()
    term_years = df_sim[term_months_col] / 12.0
    int_rate_decimal = df_sim[int_rate_col] / 100.0

    df_sim['revenue'] = int_rate_decimal * df_sim[loan_amnt_col] * term_years
    df_sim['loss'] = df_sim[loan_amnt_col]

    # Baseline: Approve All
    total_loans = len(df_sim)
    total_capital = df_sim[loan_amnt_col].sum()
    base_rev = (df_sim['revenue'] * (1 - df_sim[actual_default_col])).sum()
    base_loss = (df_sim['loss'] * df_sim[actual_default_col]).sum()
    base_net = base_rev - base_loss
    base_return_rate = base_net / total_capital if total_capital > 0 else 0.0

    rows = []
    for th in thresholds:
        approved = df_sim[df_sim[pred_prob_col] < th]
        if len(approved) == 0:
            continue

        cap = approved[loan_amnt_col].sum()
        rev = (approved['revenue'] * (1 - approved[actual_default_col])).sum()
        loss = (approved['loss'] * approved[actual_default_col]).sum()
        net = rev - loss
        ret_rate = net / cap if cap > 0 else 0.0
        improvement_pp = (ret_rate - base_return_rate) * 100.0

        # Scaled portfolio dollar value ($1B deployed)
        portfolio_return = portfolio_size * ret_rate
        baseline_return = portfolio_size * base_return_rate
        incremental_value = portfolio_return - baseline_return

        rows.append({
            'Threshold': th,
            'Loans Approved': len(approved),
            'Approval Rate': len(approved) / total_loans,
            'Default Rate': approved[actual_default_col].mean(),
            'Return Rate': ret_rate,
            'vs Baseline (pp)': improvement_pp,
            'Net Value ($1B Portfolio)': portfolio_return,
            'Incremental Gain vs Baseline': incremental_value,
        })

    return pd.DataFrame(rows)

