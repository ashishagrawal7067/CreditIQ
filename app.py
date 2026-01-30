import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Analyser | Lending Club",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — modern fintech dark theme ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #a0c4ff !important;
        font-weight: 600;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px 16px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(100,180,255,0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #a0c4ff !important;
        font-size: 0.78em !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.6em !important;
        font-weight: 800 !important;
    }

    /* ── Section headers ── */
    h1 {
        background: linear-gradient(90deg, #a0c4ff, #c77dff, #e63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4em !important;
        font-weight: 900 !important;
        padding-bottom: 8px;
    }
    h2, h3 {
        color: #a0c4ff !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(160,196,255,0.2);
        padding-bottom: 6px;
    }

    /* ── Divider ── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(160,196,255,0.3), transparent);
        margin: 24px 0;
    }

    /* ── Decision banner ── */
    .approve-banner {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        border-radius: 16px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.6em;
        font-weight: 900;
        color: white;
        letter-spacing: 0.05em;
        box-shadow: 0 8px 32px rgba(0,176,155,0.35);
        margin: 12px 0;
    }
    .decline-banner {
        background: linear-gradient(135deg, #e63946, #c1121f);
        border-radius: 16px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.6em;
        font-weight: 900;
        color: white;
        letter-spacing: 0.05em;
        box-shadow: 0 8px 32px rgba(230,57,70,0.35);
        margin: 12px 0;
    }

    /* ── Risk tier badge ── */
    .risk-badge-low {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        border-radius: 24px; padding: 6px 18px;
        font-weight: 800; font-size: 0.9em;
        display: inline-block; color: white;
    }
    .risk-badge-medium {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        border-radius: 24px; padding: 6px 18px;
        font-weight: 800; font-size: 0.9em;
        display: inline-block; color: #1a1a2e;
    }
    .risk-badge-high {
        background: linear-gradient(135deg, #f46b45, #eea849);
        border-radius: 24px; padding: 6px 18px;
        font-weight: 800; font-size: 0.9em;
        display: inline-block; color: white;
    }
    .risk-badge-veryhigh {
        background: linear-gradient(135deg, #e63946, #c1121f);
        border-radius: 24px; padding: 6px 18px;
        font-weight: 800; font-size: 0.9em;
        display: inline-block; color: white;
    }

    /* ── Info cards ── */
    .info-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
        backdrop-filter: blur(8px);
    }

    /* ── Table styling ── */
    .dataframe {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }

    /* ── Progress bars ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #a0c4ff, #c77dff) !important;
        border-radius: 8px !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 8px;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.35);
        font-size: 0.8em;
        padding: 24px 0 8px 0;
        border-top: 1px solid rgba(255,255,255,0.08);
        letter-spacing: 0.04em;
    }

    /* ── Selectbox & number input ── */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(160,196,255,0.3) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

model = load_model()

# ── Feature names in exact training order ──────────────────────────────────────
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
<div style='text-align:center; padding: 16px 0 8px 0;'>
    <div style='font-size:2.2em;'>🏦</div>
    <div style='font-size:1.1em; font-weight:800; color:#a0c4ff; 
                letter-spacing:0.06em;'>LOAN SCORER</div>
    <div style='font-size:0.75em; color:rgba(255,255,255,0.45); 
                margin-top:4px;'>Credit Risk Engine v1.0</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Loan Details")
loan_amnt   = st.sidebar.slider("Loan Amount ($)", 1000, 40000, 10000, 500)
int_rate    = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 13.0, 0.1)
term_months = st.sidebar.selectbox("Loan Term", [36, 60], index=0,
                                    format_func=lambda x: f"{x} months")
purpose     = st.sidebar.selectbox("Loan Purpose", [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business",
    "vacation", "moving", "house", "wedding",
    "renewable_energy", "educational"
], format_func=lambda x: x.replace("_", " ").title())

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Borrower Profile")
annual_inc     = st.sidebar.slider("Annual Income ($)", 20000, 300000, 65000, 1000)
fico_score     = st.sidebar.slider("FICO Score", 580, 850, 700, 5)
dti            = st.sidebar.slider("Debt-to-Income (%)", 0.0, 40.0, 15.0, 0.5)
home_ownership = st.sidebar.selectbox("Home Ownership",
                                       ["RENT", "MORTGAGE", "OWN", "OTHER"])
grade          = st.sidebar.selectbox("Loan Grade",
                                       ["A", "B", "C", "D", "E", "F", "G"],
                                       help="A = lowest risk, G = highest risk")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Credit History")
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
grade_avg_rates      = {"A": 7.49, "B": 11.72, "C": 15.25,
                         "D": 19.47, "E": 23.38, "F": 27.13, "G": 29.08}
int_rate_grade_dev   = int_rate - grade_avg_rates.get(grade, 13.0)

def build_feature_vector():
    row = {col: 0 for col in FEATURE_NAMES}
    row.update({
        'loan_amnt': loan_amnt, 'int_rate': int_rate,
        'installment': installment, 'annual_inc': annual_inc,
        'dti': dti, 'fico_mid': fico_score, 'revol_util': revol_util,
        'open_acc': open_acc, 'delinq_2yrs': delinq_2yrs, 'pub_rec': pub_rec,
        'revol_bal': revol_bal, 'total_acc': total_acc,
        'credit_age_months': credit_age_months, 'loan_to_income': loan_to_income,
        'instalment_to_income': instalment_to_income, 'delinq_flag': delinq_flag,
        'pub_rec_flag': pub_rec_flag, 'term_months': term_months,
        'int_rate_grade_dev': int_rate_grade_dev,
    })
    for g in ['B', 'C', 'D', 'E', 'F', 'G']:
        row[f'grade_{g}'] = 1 if grade == g else 0
    for p in ['credit_card','debt_consolidation','educational','home_improvement',
              'house','major_purchase','medical','moving','other',
              'renewable_energy','small_business','vacation','wedding']:
        col = f'purpose_{p}'
        if col in row:
            row[col] = 1 if purpose == p else 0
    for h in ['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT']:
        col = f'home_ownership_{h}'
        if col in row:
            row[col] = 1 if home_ownership == h else 0
    return pd.DataFrame([row])[FEATURE_NAMES]

input_df     = build_feature_vector()
default_prob = model.predict_proba(input_df)[0][1]
default_pct  = default_prob * 100

# Risk tier
if default_prob < 0.10:
    risk_tier, tier_color, tier_emoji, badge_class = \
        "Low Risk", "#00b09b", "🟢", "risk-badge-low"
elif default_prob < 0.20:
    risk_tier, tier_color, tier_emoji, badge_class = \
        "Medium Risk", "#ffd200", "🟡", "risk-badge-medium"
elif default_prob < 0.30:
    risk_tier, tier_color, tier_emoji, badge_class = \
        "High Risk", "#f46b45", "🟠", "risk-badge-high"
else:
    risk_tier, tier_color, tier_emoji, badge_class = \
        "Very High Risk", "#e63946", "🔴", "risk-badge-veryhigh"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Credit Risk Analyser")
st.markdown(
    "<p style='color:rgba(255,255,255,0.55); font-size:1.05em; margin-top:-12px;'>"
    "XGBoost model trained on <strong style='color:#a0c4ff;'>1.3M+ real Lending Club loans</strong> "
    "(2007–2018) · Gini Coefficient <strong style='color:#a0c4ff;'>0.4426</strong> · "
    "Adjust sidebar inputs to score any application instantly."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── Row 1: KPI cards ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Default Probability",  f"{default_pct:.1f}%")
c2.metric("Risk Tier",            f"{tier_emoji} {risk_tier}")
c3.metric("Loan Amount",          f"${loan_amnt:,}")
c4.metric("Monthly Payment",      f"${installment:.0f}")
c5.metric("Loan-to-Income",       f"{loan_to_income:.2f}x")

st.markdown("---")

# ── Row 2: Gauge + Decision + Signals ─────────────────────────────────────────
col_gauge, col_mid, col_signals = st.columns([1.1, 0.9, 1.2])

# Gauge
with col_gauge:
    st.markdown("### 🎯 Default Probability")

    fig, ax = plt.subplots(figsize=(5.5, 3.2), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    zones = [
        (np.linspace(np.pi, np.pi*0.75, 100), '#00b09b'),
        (np.linspace(np.pi*0.75, np.pi*0.5, 100), '#ffd200'),
        (np.linspace(np.pi*0.5, np.pi*0.25, 100), '#f46b45'),
        (np.linspace(np.pi*0.25, 0, 100), '#e63946'),
    ]
    for thetas, color in zones:
        ax.plot(np.cos(thetas), np.sin(thetas),
                color=color, linewidth=28, solid_capstyle='butt', alpha=0.9)

    # Outer ring
    ring = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(ring)*1.05, np.sin(ring)*1.05,
            color=(1,1,1,0.1), linewidth=2, alpha=0.15)

    needle_angle = np.pi * (1 - default_prob)
    ax.annotate("",
        xy=(0.72*np.cos(needle_angle), 0.72*np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=3, mutation_scale=22))
    ax.plot(0, 0, 'o', color='white', markersize=11, zorder=5)

    ax.text(0, -0.22, f"{default_pct:.1f}%",
            ha='center', va='center', fontsize=26,
            fontweight='900', color='white')
    ax.text(0, -0.46, risk_tier,
            ha='center', va='center', fontsize=10,
            fontweight='700', color=tier_color)

    # Zone labels
    ax.text(-0.95, 0.08, "Low", ha='center', fontsize=7,
            color='#00b09b', fontweight='700')
    ax.text(-0.45, 0.85, "Med", ha='center', fontsize=7,
            color='#ffd200', fontweight='700')
    ax.text(0.45, 0.85, "High", ha='center', fontsize=7,
            color='#f46b45', fontweight='700')
    ax.text(0.95, 0.08, "V.High", ha='center', fontsize=7,
            color='#e63946', fontweight='700')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.65, 1.15)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig, transparent=True)
    plt.close()

# Decision panel
with col_mid:
    st.markdown("### ⚖️ Lender Decision")
    threshold = st.slider("Approval Threshold",
                          min_value=0.05, max_value=0.50,
                          value=0.20, step=0.01, format="%.2f",
                          help="Loans below this threshold are approved")

    approved = default_prob < threshold
    banner_class = "approve-banner" if approved else "decline-banner"
    decision_text = "✅ APPROVE" if approved else "❌ DECLINE"

    st.markdown(
        f"<div class='{banner_class}'>{decision_text}</div>",
        unsafe_allow_html=True
    )

    margin = abs(default_prob - threshold) * 100
    direction = "below" if approved else "above"
    st.markdown(
        f"<div class='info-card' style='margin-top:12px; text-align:center;'>"
        f"<div style='color:rgba(255,255,255,0.5); font-size:0.8em; "
        f"text-transform:uppercase; letter-spacing:0.06em;'>Margin</div>"
        f"<div style='font-size:1.4em; font-weight:800; color:{tier_color};'>"
        f"{margin:.1f}pp {direction} threshold</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='info-card' style='margin-top:8px;'>"
        f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'>"
        f"<span style='color:rgba(255,255,255,0.5); font-size:0.82em;'>THRESHOLD</span>"
        f"<span style='font-weight:700; color:#a0c4ff;'>{threshold*100:.0f}%</span></div>"
        f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'>"
        f"<span style='color:rgba(255,255,255,0.5); font-size:0.82em;'>PRED. PROB</span>"
        f"<span style='font-weight:700; color:{tier_color};'>{default_pct:.1f}%</span></div>"
        f"<div style='display:flex; justify-content:space-between;'>"
        f"<span style='color:rgba(255,255,255,0.5); font-size:0.82em;'>RISK TIER</span>"
        f"<span class='{badge_class}'>{risk_tier}</span></div>"
        f"</div>",
        unsafe_allow_html=True
    )

# Risk signals
with col_signals:
    st.markdown("### 📡 Risk Signal Breakdown")

    signals = [
        ("FICO Score",              fico_score,        580,  850,  False, "🏅"),
        ("Debt-to-Income (%)",      dti,               0,    40,   True,  "💳"),
        ("Loan-to-Income",          round(loan_to_income,2), 0, 0.5, True, "💰"),
        ("Revolving Util (%)",      revol_util,        0,    100,  True,  "🔄"),
        ("Interest Rate (%)",       int_rate,          5,    35,   True,  "📈"),
        ("Credit Age (months)",     credit_age_months, 12,   400,  False, "📅"),
    ]

    for name, value, vmin, vmax, higher_worse, icon in signals:
        norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        if higher_worse:
            clr = '#00b09b' if norm < 0.33 else '#ffd200' if norm < 0.66 else '#e63946'
        else:
            clr = '#e63946' if norm < 0.33 else '#ffd200' if norm < 0.66 else '#00b09b'

        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"align-items:center; margin-bottom:4px;'>"
            f"<span style='font-size:0.82em; color:rgba(255,255,255,0.65);'>"
            f"{icon} {name}</span>"
            f"<span style='font-weight:700; color:{clr}; font-size:0.9em;'>{value}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        # Custom coloured progress bar via matplotlib
        fig_bar, ax_bar = plt.subplots(figsize=(3.5, 0.18))
        fig_bar.patch.set_alpha(0)
        ax_bar.set_facecolor('none')
        ax_bar.barh(0, 1, color=(1.000,1.000,1.000,0.08), height=1)
        ax_bar.barh(0, norm, color=clr, height=1)
        ax_bar.set_xlim(0, 1)
        ax_bar.axis('off')
        st.pyplot(fig_bar, transparent=True, use_container_width=True)
        plt.close()

st.markdown("---")

# ── Row 3: Portfolio charts ────────────────────────────────────────────────────
st.markdown("### 📊 Portfolio Context — How Does This Loan Compare?")

grade_defaults = {'A':0.060,'B':0.134,'C':0.224,'D':0.304,'E':0.385,'F':0.452,'G':0.499}
grade_avg_int  = {'A':7.49, 'B':11.72,'C':15.25,'D':19.47,'E':23.38,'F':27.13,'G':29.08}
grades = list(grade_defaults.keys())

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
fig3.patch.set_facecolor('#1a1a2e')

def style_ax(ax):
    ax.set_facecolor('#1a1a2e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color((1.000,1.000,1.000,0.15))
    ax.spines['bottom'].set_color((1.000,1.000,1.000,0.15))
    ax.tick_params(colors=(1.000,1.000,1.000,0.6))
    ax.xaxis.label.set_color((1.000,1.000,1.000,0.6))
    ax.yaxis.label.set_color((1.000,1.000,1.000,0.6))
    ax.title.set_color('#a0c4ff')

# Chart 1: Default rate by grade
bar_colors1 = [tier_color if g == grade else '#303060' for g in grades]
bars1 = axes3[0].bar(grades,
                      [grade_defaults[g] for g in grades],
                      color=bar_colors1, alpha=0.9,
                      edgecolor=(1,1,1,0.1), linewidth=0.5)
axes3[0].axhline(default_prob, color='#c77dff', linestyle='--',
                  linewidth=2, label=f'This loan ({default_pct:.1f}%)')
axes3[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes3[0].set_title("Default Rate by Grade", fontweight='700')
axes3[0].set_xlabel("Grade")
axes3[0].set_ylabel("Default Rate")
axes3[0].legend(fontsize=8, facecolor='#1a1a2e',
                 labelcolor='white', edgecolor=(1.000,1.000,1.000,0.2))
for bar, g in zip(bars1, grades):
    axes3[0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.005,
                   f"{grade_defaults[g]:.1%}",
                   ha='center', fontsize=8,
                   color=(1.000,1.000,1.000,0.7))
style_ax(axes3[0])

# Chart 2: Avg interest rate by grade
bar_colors2 = [tier_color if g == grade else '#303060' for g in grades]
bars2 = axes3[1].bar(grades,
                      [grade_avg_int[g] for g in grades],
                      color=bar_colors2, alpha=0.9,
                      edgecolor=(1,1,1,0.1), linewidth=0.5)
axes3[1].axhline(int_rate, color='#c77dff', linestyle='--',
                  linewidth=2, label=f'This loan ({int_rate:.1f}%)')
axes3[1].set_title("Avg Interest Rate by Grade", fontweight='700')
axes3[1].set_xlabel("Grade")
axes3[1].set_ylabel("Interest Rate (%)")
axes3[1].legend(fontsize=8, facecolor='#1a1a2e',
                 labelcolor='white', edgecolor=(1.000,1.000,1.000,0.2))
for bar, g in zip(bars2, grades):
    axes3[1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.1,
                   f"{grade_avg_int[g]:.1f}%",
                   ha='center', fontsize=8,
                   color=(1.000,1.000,1.000,0.7))
style_ax(axes3[1])

# Chart 3: Risk vs return bubble
scatter_colors = ['#00b09b','#96c93d','#ffd200','#f46b45','#e63946','#c1121f','#7b0d1e']
for i, g in enumerate(grades):
    size = 800 if g == grade else 300
    axes3[2].scatter(grade_avg_int[g], grade_defaults[g]*100,
                      s=size, color=scatter_colors[i], alpha=0.85,
                      edgecolors='white', linewidth=1.5, zorder=3)
    axes3[2].annotate(g, (grade_avg_int[g], grade_defaults[g]*100),
                       textcoords="offset points", xytext=(8, 0),
                       fontsize=10, fontweight='800',
                       color=(1.000,1.000,1.000,0.8))
axes3[2].scatter(int_rate, default_prob*100, s=500, color='#c77dff',
                  marker='*', zorder=5, label='This loan')
axes3[2].set_title("Risk vs. Return by Grade", fontweight='700')
axes3[2].set_xlabel("Avg Interest Rate (%)")
axes3[2].set_ylabel("Default Rate (%)")
axes3[2].legend(fontsize=8, facecolor='#1a1a2e',
                 labelcolor='white', edgecolor=(1.000,1.000,1.000,0.2))
axes3[2].grid(True, alpha=0.1, color='white')
style_ax(axes3[2])

plt.tight_layout()
st.pyplot(fig3, transparent=False)
plt.close()

st.markdown("---")

# ── Row 4: Model performance ───────────────────────────────────────────────────
st.markdown("### 🔬 Model Performance Summary")

mp1, mp2, mp3, mp4, mp5 = st.columns(5)
mp1.metric("Model",            "XGBoost")
mp2.metric("Gini Coefficient", "0.4426")
mp3.metric("AUC-ROC",          "0.7213")
mp4.metric("Training Loans",   "1,076,248")
mp5.metric("Business Value",   "+$17.9M / $1B")

st.markdown(
    """
    <div class='info-card'>
    <table style='width:100%; border-collapse:collapse; color:rgba(255,255,255,0.85);'>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.1);'>
            <th style='text-align:left; padding:10px; color:#a0c4ff; 
                       font-size:0.82em; text-transform:uppercase;'>Model</th>
            <th style='text-align:center; padding:10px; color:#a0c4ff; 
                       font-size:0.82em; text-transform:uppercase;'>Gini</th>
            <th style='text-align:center; padding:10px; color:#a0c4ff; 
                       font-size:0.82em; text-transform:uppercase;'>AUC-ROC</th>
            <th style='text-align:center; padding:10px; color:#a0c4ff; 
                       font-size:0.82em; text-transform:uppercase;'>PR-AUC</th>
            <th style='text-align:center; padding:10px; color:#a0c4ff; 
                       font-size:0.82em; text-transform:uppercase;'>Status</th>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
            <td style='padding:10px;'>Logistic Regression</td>
            <td style='text-align:center; padding:10px;'>0.4163</td>
            <td style='text-align:center; padding:10px;'>0.7081</td>
            <td style='text-align:center; padding:10px;'>0.3692</td>
            <td style='text-align:center;'>
                <span style='background:rgba(255,255,255,0.08); 
                             border-radius:12px; padding:3px 10px; 
                             font-size:0.78em;'>Baseline</span></td>
        </tr>
        <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
            <td style='padding:10px;'>Random Forest</td>
            <td style='text-align:center; padding:10px;'>0.4230</td>
            <td style='text-align:center; padding:10px;'>0.7115</td>
            <td style='text-align:center; padding:10px;'>0.3758</td>
            <td style='text-align:center;'>
                <span style='background:rgba(255,255,255,0.08); 
                             border-radius:12px; padding:3px 10px; 
                             font-size:0.78em;'>Evaluated</span></td>
        </tr>
        <tr>
            <td style='padding:10px; font-weight:800; color:#a0c4ff;'>
                XGBoost ★</td>
            <td style='text-align:center; padding:10px; font-weight:800; 
                       color:#00b09b;'>0.4426</td>
            <td style='text-align:center; padding:10px; font-weight:800; 
                       color:#00b09b;'>0.7213</td>
            <td style='text-align:center; padding:10px; font-weight:800; 
                       color:#00b09b;'>0.3903</td>
            <td style='text-align:center;'>
                <span style='background:linear-gradient(135deg,#00b09b,#96c93d); 
                             border-radius:12px; padding:3px 10px; 
                             font-size:0.78em; font-weight:700;'>
                    ✓ Selected</span></td>
        </tr>
    </table>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "🏦 Lending Club Credit Risk Analyser &nbsp;·&nbsp; "
    "XGBoost · 1.3M+ Loans · Gini 0.4426 &nbsp;·&nbsp; "
    "Business Simulation: +$17.9M per $1B deployed &nbsp;·&nbsp; "
    "Portfolio project — not financial advice"
    "</div>",
    unsafe_allow_html=True
)