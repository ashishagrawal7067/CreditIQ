import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lending Club Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

model = load_model()

# ── Load feature names ─────────────────────────────────────────────────────────
@st.cache_data
def load_feature_names():
    df = pd.read_parquet("data/processed/model_ready.parquet")
    return [c for c in df.columns if c != 'default']

feature_names = load_feature_names()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Loan Application Inputs
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🏦 Loan Application")
st.sidebar.markdown("Enter borrower and loan details to predict default risk.")

st.sidebar.markdown("---")
st.sidebar.subheader("Loan Details")

loan_amnt   = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 10000, 500)
int_rate    = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 13.0, 0.1)
term_months = st.sidebar.selectbox("Loan Term", [36, 60], index=0)
purpose     = st.sidebar.selectbox("Loan Purpose", [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business",
    "car", "vacation", "moving", "house", "wedding",
    "renewable_energy", "educational"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Borrower Profile")

annual_inc      = st.sidebar.slider("Annual Income ($)", 20000, 300000, 65000, 1000)
fico_score      = st.sidebar.slider("FICO Score", 580, 850, 700, 5)
dti             = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 15.0, 0.5)
home_ownership  = st.sidebar.selectbox("Home Ownership", 
                                        ["RENT", "MORTGAGE", "OWN", "OTHER"])
grade           = st.sidebar.selectbox("Loan Grade", 
                                        ["A", "B", "C", "D", "E", "F", "G"])

st.sidebar.markdown("---")
st.sidebar.subheader("Credit History")

credit_age_months   = st.sidebar.slider("Credit History Length (months)", 12, 400, 120, 6)
revol_util          = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0, 1.0)
delinq_2yrs         = st.sidebar.slider("Delinquencies (past 2 years)", 0, 10, 0)
pub_rec             = st.sidebar.slider("Public Records", 0, 5, 0)
open_acc            = st.sidebar.slider("Open Credit Accounts", 1, 40, 8)
total_acc           = st.sidebar.slider("Total Credit Accounts", 1, 80, 20)
revol_bal           = st.sidebar.slider("Revolving Balance ($)", 0, 100000, 15000, 500)
installment         = loan_amnt / term_months

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — mirror notebook 02 exactly
# ══════════════════════════════════════════════════════════════════════════════
monthly_inc          = annual_inc / 12
loan_to_income       = loan_amnt / annual_inc
instalment_to_income = installment / monthly_inc if monthly_inc > 0 else 0
delinq_flag          = 1 if delinq_2yrs > 0 else 0
pub_rec_flag         = 1 if pub_rec > 0 else 0

# Grade average interest rates (from training data — hardcoded for app)
grade_avg_rates = {
    "A": 7.49, "B": 11.72, "C": 15.25,
    "D": 19.47, "E": 23.38, "F": 27.13, "G": 29.08
}
int_rate_grade_dev = int_rate - grade_avg_rates.get(grade, 13.0)

# ── Build feature vector ───────────────────────────────────────────────────────
def build_feature_vector():
    row = {col: 0 for col in feature_names}

    # Numeric features
    row['loan_amnt']            = loan_amnt
    row['int_rate']             = int_rate
    row['installment']          = installment
    row['annual_inc']           = annual_inc
    row['dti']                  = dti
    row['fico_mid']             = fico_score
    row['revol_util']           = revol_util
    row['open_acc']             = open_acc
    row['delinq_2yrs']          = delinq_2yrs
    row['pub_rec']              = pub_rec
    row['revol_bal']            = revol_bal
    row['total_acc']            = total_acc
    row['credit_age_months']    = credit_age_months
    row['loan_to_income']       = loan_to_income
    row['instalment_to_income'] = instalment_to_income
    row['delinq_flag']          = delinq_flag
    row['pub_rec_flag']         = pub_rec_flag
    row['term_months']          = term_months
    row['int_rate_grade_dev']   = int_rate_grade_dev

    # Grade dummies (drop_first=True dropped grade_A)
    for g in ['B', 'C', 'D', 'E', 'F', 'G']:
        row[f'grade_{g}'] = 1 if grade == g else 0

    # Purpose dummies
    purpose_cols = [
        'credit_card', 'debt_consolidation', 'educational',
        'home_improvement', 'house', 'major_purchase', 'medical',
        'moving', 'other', 'renewable_energy', 'small_business',
        'vacation', 'wedding'
    ]
    for p in purpose_cols:
        col = f'purpose_{p}'
        if col in row:
            row[col] = 1 if purpose == p else 0

    # Home ownership dummies
    for h in ['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT']:
        col = f'home_ownership_{h}'
        if col in row:
            row[col] = 1 if home_ownership == h else 0

    return pd.DataFrame([row])[feature_names]

input_df     = build_feature_vector()
default_prob = model.predict_proba(input_df)[0][1]
default_pct  = default_prob * 100

# ── Risk tier ──────────────────────────────────────────────────────────────────
if default_prob < 0.10:
    risk_tier, tier_color, tier_emoji = "Low Risk", "#2ecc71", "🟢"
elif default_prob < 0.20:
    risk_tier, tier_color, tier_emoji = "Medium Risk", "#f39c12", "🟡"
elif default_prob < 0.30:
    risk_tier, tier_color, tier_emoji = "High Risk", "#e67e22", "🟠"
else:
    risk_tier, tier_color, tier_emoji = "Very High Risk", "#e74c3c", "🔴"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏦 Lending Club Credit Risk Analyser")
st.markdown(
    "Predicting loan default probability using XGBoost trained on 1.3M+ "
    "real Lending Club loans (2007–2018). Adjust inputs in the sidebar."
)

st.markdown("---")

# ── Row 1: Headline metrics ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Default Probability", f"{default_pct:.1f}%")
col2.metric("Risk Tier", f"{tier_emoji} {risk_tier}")
col3.metric("Loan Amount", f"${loan_amnt:,}")
col4.metric("Est. Monthly Payment", f"${installment:.0f}")

st.markdown("---")

# ── Row 2: Gauge + Key signals ─────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Default Probability Gauge")

    fig, ax = plt.subplots(figsize=(6, 3.5),
                           subplot_kw=dict(aspect='equal'))

    # Gauge background arcs
    theta = np.linspace(np.pi, 0, 300)
    width = 0.3

    zones = [
        (np.linspace(np.pi, np.pi*0.75, 100), '#2ecc71', 'Low (<10%)'),
        (np.linspace(np.pi*0.75, np.pi*0.5, 100), '#f39c12', 'Medium (<20%)'),
        (np.linspace(np.pi*0.5, np.pi*0.25, 100), '#e67e22', 'High (<30%)'),
        (np.linspace(np.pi*0.25, 0, 100), '#e74c3c', 'Very High (30%+)'),
    ]

    for thetas, color, _ in zones:
        ax.plot(np.cos(thetas), np.sin(thetas),
                color=color, linewidth=25, solid_capstyle='butt', alpha=0.85)

    # Needle
    needle_angle = np.pi * (1 - default_prob)
    needle_len   = 0.75
    ax.annotate("",
        xy=(needle_len * np.cos(needle_angle),
            needle_len * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                        lw=2.5, mutation_scale=20))

    ax.plot(0, 0, 'o', color='#2c3e50', markersize=10, zorder=5)

    # Centre text
    ax.text(0, -0.25, f"{default_pct:.1f}%",
            ha='center', va='center', fontsize=22,
            fontweight='bold', color=tier_color)
    ax.text(0, -0.48, risk_tier,
            ha='center', va='center', fontsize=11, color=tier_color)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.1)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Key Risk Signals")

    signals = {
        "FICO Score":              (fico_score, 580, 850, False),
        "Debt-to-Income (%)":      (dti, 0, 40, True),
        "Loan-to-Income":          (round(loan_to_income, 3), 0, 0.5, True),
        "Revolving Utilisation (%)": (revol_util, 0, 100, True),
        "Interest Rate (%)":       (int_rate, 5, 35, True),
        "Credit Age (months)":     (credit_age_months, 12, 400, False),
    }

    for signal_name, (value, vmin, vmax, higher_is_worse) in signals.items():
        normalised = (value - vmin) / (vmax - vmin)
        if higher_is_worse:
            bar_color = '#2ecc71' if normalised < 0.33 else \
                        '#f39c12' if normalised < 0.66 else '#e74c3c'
        else:
            bar_color = '#e74c3c' if normalised < 0.33 else \
                        '#f39c12' if normalised < 0.66 else '#2ecc71'

        st.markdown(f"**{signal_name}:** {value}")
        st.progress(float(np.clip(normalised, 0, 1)))

st.markdown("---")

# ── Row 3: Threshold simulator ────────────────────────────────────────────────
st.subheader("📊 Lender Decision Simulator")
st.markdown(
    "Set your approval threshold. Loans with default probability "
    "**below** the threshold are approved."
)

threshold = st.slider(
    "Approval Threshold (default probability)",
    min_value=0.05, max_value=0.50,
    value=0.20, step=0.01,
    format="%.2f"
)

decision      = "✅ APPROVE" if default_prob < threshold else "❌ DECLINE"
decision_color = "#2ecc71" if default_prob < threshold else "#e74c3c"

col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Model Decision", decision)
col_d2.metric("Predicted Default Prob", f"{default_pct:.1f}%")
col_d3.metric("Threshold Set", f"{threshold*100:.0f}%")

st.markdown("---")

# ── Row 4: Portfolio context ───────────────────────────────────────────────────
st.subheader("📈 Portfolio Context")
st.markdown(
    "How does this loan compare to Lending Club's historical "
    "default rates by grade?"
)

grade_defaults = {
    'A': 0.060, 'B': 0.134, 'C': 0.224,
    'D': 0.304, 'E': 0.385, 'F': 0.452, 'G': 0.499
}

fig2, ax2 = plt.subplots(figsize=(10, 4))
grades  = list(grade_defaults.keys())
rates   = list(grade_defaults.values())
colors  = ['#2ecc71' if g != grade else tier_color for g in grades]
bars    = ax2.bar(grades, rates, color=colors, alpha=0.8)

ax2.axhline(default_prob, color='#2c3e50', linestyle='--',
            linewidth=2, label=f'This loan ({default_pct:.1f}%)')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.set_title("Historical Default Rate by Grade vs. This Loan's Predicted Risk")
ax2.set_xlabel("Loan Grade")
ax2.set_ylabel("Default Rate")
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, rate in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{rate:.1%}", ha='center', fontsize=9)

plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
    Built with XGBoost · Trained on 1.3M Lending Club loans (2007–2018) · 
    Gini Coefficient: 0.4426 · For portfolio demonstration purposes
    </div>
    """,
    unsafe_allow_html=True
)