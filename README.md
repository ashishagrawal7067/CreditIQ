import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Lending Risk Analyser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 40%, #0a1220 100%);
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #091428 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}
[data-testid="stSidebar"] * {
    color: #cbd5e0 !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #90cdf4 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(154,117,234,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 20px 24px !important;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,179,237,0.5);
}
[data-testid="stMetricLabel"] {
    color: #90cdf4 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(99,179,237,0.12) !important;
    margin: 1.5rem 0 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #4299e1, #9a75ea) !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4299e1, #9a75ea) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #90cdf4;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(66,153,225,0.15) 0%, rgba(154,117,234,0.15) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(20px);
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #63b3ed, #9a75ea, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem !important;
}
.hero p {
    color: #a0aec0;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 99px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Info box ── */
.info-box {
    background: rgba(99,179,237,0.07);
    border-left: 3px solid #4299e1;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #a0aec0;
    line-height: 1.6;
}

/* ── Table ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
th {
    background: rgba(66,153,225,0.15) !important;
    color: #90cdf4 !important;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}
td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e0;
}
tr:hover td { background: rgba(99,179,237,0.05); }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

model = load_model()

# ── Feature names — exact order from model training ───────────────────────────
FEATURE_NAMES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'fico_mid', 'revol_util', 'open_acc', 'delinq_2yrs', 'pub_rec',
    'revol_bal', 'total_acc', 'credit_age_months', 'loan_to_income',
    'instalment_to_income', 'delinq_flag', 'pub_rec_flag', 'term_months',
    'int_rate_grade_dev', 'grade_B', 'grade_C', 'grade_D', 'grade_E',
    'grade_F', 'grade_G', 'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_educational', 'purpose_home_improvement', 'purpose_house',
    'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
    'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
    'purpose_vacation', 'purpose_wedding', 'home_ownership_MORTGAGE',
    'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN',
    'home_ownership_RENT'
]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding: 16px 0 8px;'>
    <span style='font-family:Syne,sans-serif; font-size:1.3rem; 
    font-weight:800; background:linear-gradient(135deg,#63b3ed,#9a75ea);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    CreditIQ
    </span><br>
    <span style='font-size:0.72rem; color:#718096; letter-spacing:0.1em; 
    text-transform:uppercase;'>Risk Analyser</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Loan Details</p>',
                    unsafe_allow_html=True)

loan_amnt   = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 10000, 500)
int_rate    = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 13.0, 0.1)
term_months = st.sidebar.selectbox("Loan Term", [36, 60], index=0)
purpose     = st.sidebar.selectbox("Loan Purpose", [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business",
    "vacation", "moving", "house", "wedding",
    "renewable_energy", "educational"
])

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Borrower Profile</p>',
                    unsafe_allow_html=True)

annual_inc     = st.sidebar.slider("Annual Income ($)", 20000, 300000, 65000, 1000)
fico_score     = st.sidebar.slider("FICO Score", 580, 850, 700, 5)
dti            = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 15.0, 0.5)
home_ownership = st.sidebar.selectbox("Home Ownership",
                                       ["RENT", "MORTGAGE", "OWN", "OTHER"])
grade          = st.sidebar.selectbox("Loan Grade",
                                       ["A", "B", "C", "D", "E", "F", "G"])

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Credit History</p>',
                    unsafe_allow_html=True)

credit_age_months = st.sidebar.slider("Credit History (months)", 12, 400, 120, 6)
revol_util        = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.0, 1.0)
delinq_2yrs       = st.sidebar.slider("Delinquencies (past 2 yrs)", 0, 10, 0)
pub_rec           = st.sidebar.slider("Public Records", 0, 5, 0)
open_acc          = st.sidebar.slider("Open Credit Accounts", 1, 40, 8)
total_acc         = st.sidebar.slider("Total Credit Accounts", 1, 80, 20)
revol_bal         = st.sidebar.slider("Revolving Balance ($)", 0, 100000, 15000, 500)
installment       = loan_amnt / term_months

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
monthly_inc          = annual_inc / 12
loan_to_income       = loan_amnt / annual_inc
instalment_to_income = installment / monthly_inc if monthly_inc > 0 else 0
delinq_flag          = 1 if delinq_2yrs > 0 else 0
pub_rec_flag         = 1 if pub_rec > 0 else 0

grade_avg_rates = {
    "A": 7.49, "B": 11.72, "C": 15.25,
    "D": 19.47, "E": 23.38, "F": 27.13, "G": 29.08
}
int_rate_grade_dev = int_rate - grade_avg_rates.get(grade, 13.0)

def build_feature_vector():
    row = {col: 0 for col in FEATURE_NAMES}
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
    for g in ['B', 'C', 'D', 'E', 'F', 'G']:
        row[f'grade_{g}'] = 1 if grade == g else 0
    for p in ['credit_card','debt_consolidation','educational',
              'home_improvement','house','major_purchase','medical',
              'moving','other','renewable_energy','small_business',
              'vacation','wedding']:
        col = f'purpose_{p}'
        if col in row:
            row[col] = 1 if purpose == p else 0
    for h in ['MORTGAGE','NONE','OTHER','OWN','RENT']:
        col = f'home_ownership_{h}'
        if col in row:
            row[col] = 1 if home_ownership == h else 0
    return pd.DataFrame([row])[FEATURE_NAMES]

input_df     = build_feature_vector()
default_prob = model.predict_proba(input_df)[0][1]
default_pct  = default_prob * 100

# Risk tier config
TIERS = [
    (0.10, "Low Risk",       "#48bb78", "#1a4731", "🟢"),
    (0.20, "Medium Risk",    "#ecc94b", "#4a3800", "🟡"),
    (0.30, "High Risk",      "#ed8936", "#4a2000", "🟠"),
    (1.00, "Very High Risk", "#fc8181", "#4a1515", "🔴"),
]
for threshold_t, label, color, bg, emoji in TIERS:
    if default_prob < threshold_t:
        risk_tier, tier_color, tier_bg, tier_emoji = label, color, bg, emoji
        break

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown(f"""
<div class="hero">
    <h1>CreditIQ — Loan Default Risk Analyser</h1>
    <p>Production-grade credit risk scoring powered by XGBoost · Trained on
    <strong style="color:#90cdf4">1.3M+ real Lending Club loans</strong>
    (2007–2018) · Gini Coefficient <strong style="color:#9a75ea">0.4426</strong>
    · Adjust inputs in the sidebar to score any application instantly.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Default Probability",  f"{default_pct:.1f}%")
k2.metric("Risk Tier",            f"{tier_emoji} {risk_tier}")
k3.metric("Loan Amount",          f"${loan_amnt:,}")
k4.metric("Monthly Payment",      f"${installment:.0f}")
k5.metric("Loan-to-Income",       f"{loan_to_income:.2f}x")

st.markdown("---")

# ── Gauge + Signals ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<p class="section-header">Default Probability Gauge</p>',
                unsafe_allow_html=True)

    # Dark themed gauge
    fig, ax = plt.subplots(figsize=(6, 3.8), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0d1528')
    ax.set_facecolor('#0d1528')

    zones = [
        (np.linspace(np.pi, np.pi * 0.75, 100), '#48bb78'),
        (np.linspace(np.pi * 0.75, np.pi * 0.5, 100), '#ecc94b'),
        (np.linspace(np.pi * 0.5, np.pi * 0.25, 100), '#ed8936'),
        (np.linspace(np.pi * 0.25, 0, 100), '#fc8181'),
    ]
    for thetas, color in zones:
        ax.plot(np.cos(thetas), np.sin(thetas),
                color=color, linewidth=28, solid_capstyle='butt', alpha=0.9)

    # Inner dark ring for depth
    for thetas, _ in zones:
        ax.plot(np.cos(thetas) * 0.72, np.sin(thetas) * 0.72,
                color='#0d1528', linewidth=10, solid_capstyle='butt')

    needle_angle = np.pi * (1 - default_prob)
    ax.annotate("",
        xy=(0.68 * np.cos(needle_angle), 0.68 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#ffffff",
                        lw=2.5, mutation_scale=18))
    ax.plot(0, 0, 'o', color='#ffffff', markersize=9, zorder=5)

    # Zone labels
    for angle, label in [(np.pi * 0.875, 'LOW'), (np.pi * 0.625, 'MED'),
                          (np.pi * 0.375, 'HIGH'), (np.pi * 0.125, 'V.HIGH')]:
        ax.text(0.88 * np.cos(angle), 0.88 * np.sin(angle), label,
                ha='center', va='center', fontsize=6.5,
                color='#4a5568', fontweight='bold')

    ax.text(0, -0.18, f"{default_pct:.1f}%",
            ha='center', va='center', fontsize=26,
            fontweight='bold', color=tier_color,
            fontfamily='DejaVu Sans')
    ax.text(0, -0.42, risk_tier,
            ha='center', va='center', fontsize=10,
            color=tier_color, fontweight='bold')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.65, 1.05)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    st.pyplot(fig)
    plt.close()

    # Decision box
    threshold_app = st.slider(
        "Approval Threshold",
        min_value=0.05, max_value=0.50,
        value=0.20, step=0.01,
        format="%.0%%"
    )
    approved = default_prob < threshold_app
    dec_color = "#48bb78" if approved else "#fc8181"
    dec_label = "✅ APPROVE" if approved else "❌ DECLINE"
    margin    = abs(default_prob - threshold_app) * 100

    st.markdown(f"""
    <div style='background:{"rgba(72,187,120,0.1)" if approved else "rgba(252,129,129,0.1)"};
    border:1px solid {dec_color}; border-radius:14px; padding:16px 20px; margin-top:8px;
    text-align:center;'>
        <div style='font-family:Syne,sans-serif; font-size:1.5rem;
        font-weight:800; color:{dec_color};'>{dec_label}</div>
        <div style='color:#a0aec0; font-size:0.8rem; margin-top:4px;'>
        {margin:.1f}pp {"below" if approved else "above"} threshold
        ({threshold_app*100:.0f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<p class="section-header">Risk Signal Dashboard</p>',
                unsafe_allow_html=True)

    signals = [
        ("FICO Score",              fico_score,        580,  850,  False, "Higher is safer"),
        ("Debt-to-Income (%)",      dti,               0,    40,   True,  "Lower is safer"),
        ("Loan-to-Income",          round(loan_to_income,3), 0, 0.5, True, "Lower is safer"),
        ("Revolving Utilisation %", revol_util,        0,    100,  True,  "Lower is safer"),
        ("Interest Rate (%)",       int_rate,          5,    35,   True,  "Lower is safer"),
        ("Credit Age (months)",     credit_age_months, 12,   400,  False, "Longer is safer"),
    ]

    for name, value, vmin, vmax, higher_worse, hint in signals:
        norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        if higher_worse:
            c = '#48bb78' if norm < 0.33 else '#ecc94b' if norm < 0.66 else '#fc8181'
        else:
            c = '#fc8181' if norm < 0.33 else '#ecc94b' if norm < 0.66 else '#48bb78'

        st.markdown(f"""
        <div style='margin-bottom:14px;'>
            <div style='display:flex; justify-content:space-between;
            align-items:baseline; margin-bottom:4px;'>
                <span style='font-size:0.82rem; color:#a0aec0;
                font-weight:500;'>{name}</span>
                <span style='font-family:Syne,sans-serif; font-weight:700;
                color:{c}; font-size:0.95rem;'>{value}</span>
            </div>
            <div style='background:rgba(255,255,255,0.06); border-radius:99px;
            height:7px; overflow:hidden;'>
                <div style='width:{norm*100:.1f}%; height:100%;
                background:linear-gradient(90deg, {c}aa, {c});
                border-radius:99px; transition:width 0.3s ease;'></div>
            </div>
            <div style='font-size:0.7rem; color:#4a5568;
            margin-top:2px;'>{hint}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Portfolio Context ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Portfolio Context</p>',
            unsafe_allow_html=True)

grade_defaults = {'A':0.060,'B':0.134,'C':0.224,'D':0.304,'E':0.385,'F':0.452,'G':0.499}
grade_avg_int  = {'A':7.49,'B':11.72,'C':15.25,'D':19.47,'E':23.38,'F':27.13,'G':29.08}
grades_list    = list(grade_defaults.keys())

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4))
for ax in axes3:
    ax.set_facecolor('#0d1528')
    fig3.patch.set_facecolor('#0d1528')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#718096')
    ax.yaxis.label.set_color('#718096')
    ax.xaxis.label.set_color('#718096')
    ax.title.set_color('#90cdf4')

# Default rate
bar_colors = ['#9a75ea' if g == grade else '#1e3a5f' for g in grades_list]
bars1 = axes3[0].bar(grades_list,
                      [grade_defaults[g] for g in grades_list],
                      color=bar_colors, alpha=0.9, width=0.6)
axes3[0].axhline(default_prob, color='#fc8181', linestyle='--',
                  linewidth=2, label=f'This loan ({default_pct:.1f}%)')
axes3[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes3[0].set_title("Historical Default Rate by Grade", fontsize=11, pad=10)
axes3[0].set_xlabel("Loan Grade", fontsize=9)
axes3[0].set_ylabel("Default Rate", fontsize=9)
axes3[0].legend(fontsize=8, facecolor='#0d1528',
                labelcolor='#a0aec0', edgecolor='#2d3748')
for bar, g in zip(bars1, grades_list):
    axes3[0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.006,
                   f"{grade_defaults[g]:.1%}",
                   ha='center', fontsize=8.5,
                   color='#9a75ea' if g == grade else '#4a5568',
                   fontweight='bold' if g == grade else 'normal')

# Avg interest rate
bar_colors2 = ['#63b3ed' if g == grade else '#1e3a5f' for g in grades_list]
bars2 = axes3[1].bar(grades_list,
                      [grade_avg_int[g] for g in grades_list],
                      color=bar_colors2, alpha=0.9, width=0.6)
axes3[1].axhline(int_rate, color='#fc8181', linestyle='--',
                  linewidth=2, label=f'This loan ({int_rate:.1f}%)')
axes3[1].set_title("Average Interest Rate by Grade", fontsize=11, pad=10)
axes3[1].set_xlabel("Loan Grade", fontsize=9)
axes3[1].set_ylabel("Interest Rate (%)", fontsize=9)
axes3[1].legend(fontsize=8, facecolor='#0d1528',
                labelcolor='#a0aec0', edgecolor='#2d3748')
for bar, g in zip(bars2, grades_list):
    axes3[1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.2,
                   f"{grade_avg_int[g]:.1f}%",
                   ha='center', fontsize=8.5,
                   color='#63b3ed' if g == grade else '#4a5568',
                   fontweight='bold' if g == grade else 'normal')

plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")

# ── Model Performance ──────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Model Performance</p>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Model",   "XGBoost")
m2.metric("Gini Coefficient", "0.4426")
m3.metric("AUC-ROC",          "0.7213")
m4.metric("Business Value",   "+$17.9M / $1B")

st.markdown("""
| Model | AUC-ROC | Gini | PR-AUC | Training Time |
|-------|---------|------|--------|---------------|
| Logistic Regression | 0.7081 | 0.4163 | 0.3692 | 12.4s |
| Random Forest | 0.7115 | 0.4230 | 0.3758 | 451.9s |
| **XGBoost ✓** | **0.7213** | **0.4426** | **0.3903** | **93.6s** |
""")

st.markdown("""
<div class="info-box">
    <strong style="color:#90cdf4">Gini Coefficient</strong> is the industry-standard 
    metric in consumer credit risk (= 2×AUC−1). A Gini of 0 = random; 1.0 = perfect. 
    Industry production models typically achieve 0.40–0.65. This model's Gini of 
    <strong style="color:#9a75ea">0.4426</strong> sits comfortably within that range, 
    trained on 1.3M+ real loans across an 11-year period including the 2008 financial crisis.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 10px; 
color:#4a5568; font-size:0.78rem; letter-spacing:0.04em;'>
    CreditIQ · Built with XGBoost · 1.3M+ Lending 
    Club Loans (2007–2018) · 
    Gini 0.4426 · Business Simulation: +$17.9M per $1B Deployed · 
    For Portfolio Demonstration Purposes Only
</div>
""", unsafe_allow_html=True)