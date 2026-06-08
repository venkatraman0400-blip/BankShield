"""
BankShield — Fraud Detection Web App
Streamlit UI: transaction input -> Random Forest prediction -> SHAP explainability
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURE_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "type_encoded",
    "balanceOrigDiff", "balanceDestDiff",
    "errorBalanceOrig", "errorBalanceDest",
    "isHighAmount", "zeroBalanceOrig", "zeroBalanceDest",
    "amountToBalance", "amountToDestBalance", "accountDrained",
]

FEATURE_LABELS = {
    "step":               "Transaction Step (Hour)",
    "amount":             "Transaction Amount ($)",
    "oldbalanceOrg":      "Origin — Opening Balance ($)",
    "newbalanceOrig":     "Origin — Closing Balance ($)",
    "oldbalanceDest":     "Destination — Opening Balance ($)",
    "newbalanceDest":     "Destination — Closing Balance ($)",
    "type_encoded":       "Transaction Type",
    "balanceOrigDiff":    "Origin Balance Change ($)",
    "balanceDestDiff":    "Destination Balance Change ($)",
    "errorBalanceOrig":   "Balance Error — Origin ($)",
    "errorBalanceDest":   "Balance Error — Destination ($)",
    "isHighAmount":       "High Amount Flag",
    "zeroBalanceOrig":    "Origin Was Zero",
    "zeroBalanceDest":    "Destination Was Zero",
    "amountToBalance":    "Amount / Origin Balance Ratio",
    "amountToDestBalance":"Amount / Dest Balance Ratio",
    "accountDrained":     "Account Drained to Zero",
}

MODEL_PATH = Path("models/model.pkl")
FEAT_PATH  = Path("models/feature_names.pkl")
LE_PATH    = Path("models/label_encoder.pkl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankShield — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark theme) ───────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #58a6ff; }

div[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.78rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important; font-size: 1.4rem; font-weight: 700;
}

.fraud-banner {
    background: linear-gradient(135deg, #3d0000, #7d0000);
    border: 1px solid #f85149; border-radius: 10px;
    padding: 20px 28px; text-align: center;
}
.legit-banner {
    background: linear-gradient(135deg, #003d1a, #006b2e);
    border: 1px solid #3fb950; border-radius: 10px;
    padding: 20px 28px; text-align: center;
}
.banner-verdict { font-size: 2.0rem; font-weight: 800; letter-spacing: 2px; }
.banner-sub     { font-size: 1.1rem; margin-top: 6px; opacity: 0.85; }
.risk-label     { font-size: 0.85rem; color: #8b949e; margin-bottom: 2px; }
.risk-score     { font-size: 3.5rem; font-weight: 900; line-height: 1.1; }
.section-header {
    border-left: 3px solid #58a6ff; padding-left: 10px;
    color: #c9d1d9; font-size: 1.05rem; font-weight: 600; margin-bottom: 4px;
}
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── Feature engineering (must match train.py exactly) ─────────────────────────
def engineer_features(df: pd.DataFrame, le) -> pd.DataFrame:
    df = df.copy()
    df["balanceOrigDiff"]     = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDestDiff"]     = df["newbalanceDest"] - df["oldbalanceDest"]
    df["errorBalanceOrig"]    = df["amount"] - (df["oldbalanceOrg"] - df["newbalanceOrig"])
    df["errorBalanceDest"]    = df["amount"] - (df["newbalanceDest"] - df["oldbalanceDest"])
    df["amountToBalance"]     = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amountToDestBalance"] = df["amount"] / (df["oldbalanceDest"] + 1)
    df["isHighAmount"]        = 0   # single-row: flag is 0 by default (no distribution to quantile)
    df["zeroBalanceOrig"]     = (df["oldbalanceOrg"] == 0).astype(int)
    df["zeroBalanceDest"]     = (df["oldbalanceDest"] == 0).astype(int)
    df["accountDrained"]      = (df["newbalanceOrig"] == 0).astype(int)
    df["type_encoded"]        = le.transform(df["type"])
    return df


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BankShield model...")
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None, None
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    le       = joblib.load(LE_PATH)
    return model, features, le


@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Shield BankShield")
    st.markdown("**AI-powered fraud detection** for mobile financial transactions.")
    st.markdown("---")

    st.markdown("### Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("AUC-ROC",   "1.00")
    c2.metric("F1 Score",  "0.998")
    c3, c4 = st.columns(2)
    c3.metric("Precision", "1.00")
    c4.metric("Recall",    "0.996")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
- **Model**: Random Forest (200 trees)
- **Training**: 50,000 synthetic transactions
- **Features**: 17 engineered features
- **Explainability**: SHAP values
- **Fraud types**: CASH_OUT & TRANSFER only
""")
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#8b949e;'>Built with Streamlit · "
        "Powered by Random Forest & SHAP</p>",
        unsafe_allow_html=True,
    )


# ── Main title ────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ BankShield — Fraud Detection")
st.markdown(
    "Enter transaction details below and click **Analyse Transaction** "
    "for an instant fraud risk assessment with AI explainability."
)
st.markdown("---")

# ── Load artifacts ────────────────────────────────────────────────────────────
model, feature_names, le = load_artifacts()

if model is None:
    st.error(
        "**Model not found.** Run the training pipeline first:\n\n"
        "```bash\npython generate_data.py\npython train.py\n```"
    )
    st.stop()

explainer = get_explainer(model)

# ── Transaction input form ────────────────────────────────────────────────────
st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)

with st.form("transaction_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Transaction Info**")
        txn_type = st.selectbox(
            "Transaction Type",
            options=TRANSACTION_TYPES,
            index=1,
            help="Fraud occurs almost exclusively in CASH_OUT and TRANSFER transactions.",
        )
        step = st.slider(
            "Step (transaction hour)", min_value=1, max_value=744, value=1,
            help="1 hour unit of time — 744 steps = 31 days"
        )
        amount = st.number_input(
            "Amount ($)", min_value=0.01, max_value=10_000_000.0,
            value=1000.0, step=0.01, format="%.2f"
        )

    with col_b:
        st.markdown("**Account Balances**")
        old_orig = st.number_input(
            "Origin — Opening Balance ($)", min_value=0.0, max_value=50_000_000.0,
            value=5000.0, step=0.01, format="%.2f"
        )
        new_orig = st.number_input(
            "Origin — Closing Balance ($)", min_value=0.0, max_value=50_000_000.0,
            value=4000.0, step=0.01, format="%.2f"
        )
        old_dest = st.number_input(
            "Destination — Opening Balance ($)", min_value=0.0, max_value=50_000_000.0,
            value=0.0, step=0.01, format="%.2f"
        )
        new_dest = st.number_input(
            "Destination — Closing Balance ($)", min_value=0.0, max_value=50_000_000.0,
            value=0.0, step=0.01, format="%.2f"
        )

    st.markdown("")
    submitted = st.form_submit_button(
        "🔍 Analyse Transaction", use_container_width=True, type="primary"
    )

# ── Prediction & output ───────────────────────────────────────────────────────
if submitted:
    raw = pd.DataFrame([{
        "step":           step,
        "type":           txn_type,
        "amount":         amount,
        "oldbalanceOrg":  old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }])

    input_data = engineer_features(raw, le)[FEATURE_COLS]

    fraud_prob = float(model.predict_proba(input_data)[0, 1])
    risk_score = int(round(fraud_prob * 100))
    is_fraud   = fraud_prob >= 0.5
    confidence = fraud_prob if is_fraud else (1 - fraud_prob)

    st.markdown("---")
    st.markdown('<p class="section-header">Analysis Result</p>', unsafe_allow_html=True)

    res_col, score_col = st.columns([2, 1])

    with res_col:
        if is_fraud:
            st.markdown(f"""
<div class="fraud-banner">
  <div class="banner-verdict" style="color:#f85149;">&#9888; FRAUDULENT TRANSACTION</div>
  <div class="banner-sub" style="color:#ffa198;">Confidence: <strong>{confidence:.1%}</strong></div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="legit-banner">
  <div class="banner-verdict" style="color:#3fb950;">&#10003; LEGITIMATE TRANSACTION</div>
  <div class="banner-sub" style="color:#7ee787;">Confidence: <strong>{confidence:.1%}</strong></div>
</div>""", unsafe_allow_html=True)

    with score_col:
        risk_color = "#f85149" if risk_score >= 70 else ("#e3b341" if risk_score >= 40 else "#3fb950")
        st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
            padding:20px; text-align:center;">
  <div class="risk-label">FRAUD RISK SCORE</div>
  <div class="risk-score" style="color:{risk_color};">{risk_score}</div>
  <div class="risk-label">out of 100</div>
</div>""", unsafe_allow_html=True)

    # ── Risk bar ──────────────────────────────────────────────────────────────
    st.markdown("")
    bar_col = st.columns([1, 6, 1])[1]
    with bar_col:
        fig_bar, ax_bar = plt.subplots(figsize=(7, 0.55))
        fig_bar.patch.set_facecolor("#0e1117")
        ax_bar.set_facecolor("#0e1117")
        ax_bar.imshow(
            np.linspace(0, 1, 300).reshape(1, -1),
            aspect="auto", cmap="RdYlGn_r",
            extent=[0, 100, 0, 1], vmin=0, vmax=1
        )
        ax_bar.axvline(x=risk_score, color="white", linewidth=2.5)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Risk Score ->", color="#8b949e", fontsize=8)
        ax_bar.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#30363d")
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # ── Engineered feature summary ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Computed Features</p>', unsafe_allow_html=True)
    feat_row = input_data.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Balance Error (Origin)",  f"${feat_row['errorBalanceOrig']:,.2f}")
    m2.metric("Balance Error (Dest)",    f"${feat_row['errorBalanceDest']:,.2f}")
    m3.metric("Amount / Origin Ratio",   f"{feat_row['amountToBalance']:.3f}")
    m4.metric("Account Drained",         "Yes" if feat_row["accountDrained"] else "No")

    # ── SHAP feature importance ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">AI Explainability — Why This Decision?</p>',
                unsafe_allow_html=True)
    st.markdown(
        "SHAP values show how each feature pushed the prediction toward "
        "**Fraud** (red) or **Legitimate** (blue)."
    )

    shap_values = explainer(input_data)
    sv = shap_values[0]

    readable = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    vals      = sv.values
    order     = np.argsort(np.abs(vals))[::-1]

    sorted_vals   = vals[order]
    sorted_labels = [readable[i] for i in order]
    colors        = ["#f85149" if v > 0 else "#58a6ff" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    y_pos = range(len(sorted_vals) - 1, -1, -1)
    ax.barh(list(y_pos), sorted_vals[::-1], color=colors[::-1], height=0.65, zorder=2)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_labels[::-1], color="#c9d1d9", fontsize=8)
    ax.set_xlabel("SHAP Value (impact on fraud probability)", color="#8b949e", fontsize=9)
    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--", zorder=1)
    ax.tick_params(axis="x", colors="#8b949e", labelsize=8)
    ax.grid(axis="x", color="#30363d", linestyle="--", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title("Feature Contribution to Fraud Prediction", color="#c9d1d9", fontsize=11, pad=12)

    ax.legend(
        handles=[
            plt.Rectangle((0,0),1,1, fc="#f85149", label="Increases Fraud Risk"),
            plt.Rectangle((0,0),1,1, fc="#58a6ff", label="Decreases Fraud Risk"),
        ],
        loc="lower right", facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#c9d1d9", fontsize=8
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Raw values table ──────────────────────────────────────────────────────
    with st.expander("View Raw Feature Values & SHAP Scores"):
        display_df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
            "Value":      [float(input_data[f].iloc[0]) for f in FEATURE_COLS],
            "SHAP Value": [round(float(sv.values[i]), 5) for i in range(len(FEATURE_COLS))],
        })
        st.dataframe(
            display_df.style
                .format({"Value": "{:.4f}", "SHAP Value": "{:+.5f}"})
                .map(
                    lambda v: "color: #f85149" if isinstance(v, float) and v > 0
                    else ("color: #58a6ff" if isinstance(v, float) and v < 0 else ""),
                    subset=["SHAP Value"],
                ),
            use_container_width=True,
            hide_index=True,
        )

else:
    # ── Placeholder ───────────────────────────────────────────────────────────
    st.info(
        "Fill in the transaction details above and click "
        "**Analyse Transaction** to get an instant fraud risk assessment."
    )

    st.markdown('<p class="section-header">Key Fraud Signals in This Dataset</p>',
                unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.markdown("""
**Transaction Type**
- Only CASH_OUT and TRANSFER carry fraud risk
- PAYMENT, DEBIT, CASH_IN = always legitimate
""")
    with hc2:
        st.markdown("""
**Balance Anomalies**
- Origin account drained to exactly zero
- Balance arithmetic doesn't add up (error features)
- Origin had zero balance before transaction
""")
    with hc3:
        st.markdown("""
**Amount Patterns**
- Amount equals the full opening balance
- High amount relative to origin balance
- Destination balance stays at zero after transfer
""")
