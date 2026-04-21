# =============================================================
#  Fraud Detection — Streamlit Demo App
#  Run: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "fraud_model.pkl"
DATA_PATH  = "creditcard.csv"

def download_dataset():
    """Download creditcard.csv from Kaggle if not present."""
    if os.path.exists(DATA_PATH):
        return True
    try:
        import kaggle
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY']      = st.secrets["KAGGLE_KEY"]
        with st.spinner("📥 Downloading dataset from Kaggle (first time only)..."):
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "mlg-ulb/creditcardfraud", path=".", unzip=True
            )
        return os.path.exists(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        return False

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #3a3a6e;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(101, 17, 203, 0.4);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: #68d391 !important; }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #a78bfa;
        letter-spacing: 0.5px;
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Result cards ── */
    .result-fraud {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(255, 65, 108, 0.4);
        margin: 1rem 0;
    }
    .result-legit {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(56, 239, 125, 0.3);
        margin: 1rem 0;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(101,17,203,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(101,17,203,0.6);
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6a11cb, #ff416c) !important;
        border-radius: 10px;
    }

    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #6a11cb !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.35);
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem 0;
    }
    .footer a { color: #7c6fcd; text-decoration: none; }
    .footer a:hover { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD & TRAIN (cached so it only runs once)
# ------------------------------------------------------------
@st.cache_resource
def load_and_train():
    # ── Load from disk if already trained ──
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return (saved['rf'], saved['scaler'], saved['X_test'], saved['y_test'],
                saved['y_proba'], saved['fpr'], saved['tpr'], saved['roc_auc'],
                saved['feat_imp'], saved['df'])

    # ── First run: train and save ──
    df = pd.read_csv(DATA_PATH)

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)

    y_proba = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    feat_imp = pd.Series(rf.feature_importances_, index=X.columns)

    joblib.dump({
        'rf': rf, 'scaler': scaler, 'X_test': X_test, 'y_test': y_test,
        'y_proba': y_proba, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'feat_imp': feat_imp, 'df': df
    }, MODEL_PATH)

    return rf, scaler, X_test, y_test, y_proba, fpr, tpr, roc_auc, feat_imp, df

# ------------------------------------------------------------
# HERO BANNER
# ------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🛡️ Credit Card Fraud Detection</h1>
    <p>AI-powered transaction analysis using Random Forest + SMOTE · Real-time fraud scoring</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR — MANUAL TRANSACTION INPUT
# ------------------------------------------------------------
st.sidebar.markdown("## 🧾 Test a Transaction")
st.sidebar.markdown("Simulate a transaction and get an instant fraud prediction.")
st.sidebar.markdown("---")

amount = st.sidebar.slider("💵 Transaction Amount (USD)", 0.0, 5000.0, 120.0, 5.0)
time   = st.sidebar.slider("⏱️ Time Elapsed (seconds)", 0, 172800, 50000, 1000)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 V-Features** *(PCA-transformed)*")
st.sidebar.caption("Leave at 0 for a typical transaction. Drag to simulate anomalies.")
v14 = st.sidebar.slider("V14 — most predictive", -20.0, 10.0, 0.0, 0.1)
v17 = st.sidebar.slider("V17", -20.0, 10.0, 0.0, 0.1)
v12 = st.sidebar.slider("V12", -20.0, 10.0, 0.0, 0.1)
v10 = st.sidebar.slider("V10", -20.0, 10.0, 0.0, 0.1)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔎 Analyse Transaction", use_container_width=True)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
with st.spinner("⚙️ Loading model... (first run trains on 284,807 transactions, ~1 min)"):
    try:
        if not download_dataset():
            st.stop()
        rf, scaler, X_test, y_test, y_proba, fpr, tpr, roc_auc, feat_imp, df = load_and_train()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

if not model_loaded:
    st.error("⚠️ `creditcard.csv` not found.")
    st.markdown("[📥 Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.stop()

# ------------------------------------------------------------
# PREDICTION PANEL
# ------------------------------------------------------------
if predict_btn:
    st.markdown('<div class="section-title">🔎 Prediction Result</div>', unsafe_allow_html=True)

    sample = np.zeros(30)
    sample[13] = v14
    sample[16] = v17
    sample[11] = v12
    sample[9]  = v10
    sample[28] = scaler.transform([[amount]])[0][0]
    sample[29] = (time - 94813) / 47488

    proba = rf.predict_proba([sample])[0][1]
    is_fraud = proba >= 0.5

    col1, col2, col3 = st.columns(3)
    col1.metric("Verdict", "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE")
    col2.metric("Fraud Probability", f"{proba*100:.1f}%")
    col3.metric("Amount", f"${amount:,.2f}")

    st.progress(float(proba))

    if is_fraud:
        st.markdown(f'<div class="result-fraud">🚨 Transaction flagged as <strong>FRAUDULENT</strong> — {proba*100:.1f}% confidence. Recommend blocking immediately.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-legit">✅ Transaction appears <strong>LEGITIMATE</strong> — only {proba*100:.1f}% fraud probability. Safe to proceed.</div>', unsafe_allow_html=True)

    st.divider()

# ------------------------------------------------------------
# METRICS OVERVIEW
# ------------------------------------------------------------
st.markdown('<div class="section-title">📊 Model Performance Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 ROC-AUC Score",       f"{roc_auc:.4f}",  "Random Forest")
col2.metric("📁 Total Transactions",  "284,807",          "Kaggle dataset")
col3.metric("🚨 Fraud Cases",         "492",              "0.17% of all txns")
col4.metric("⚖️ Imbalance Fix",       "SMOTE",            "Synthetic oversampling")

# ------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------
st.markdown('<div class="section-title">📈 ROC Curve & Feature Importance</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#1a1a2e')

for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#a0aec0')
    ax.xaxis.label.set_color('#a0aec0')
    ax.yaxis.label.set_color('#a0aec0')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a6e')

# ROC Curve
axes[0].plot(fpr, tpr, color='#a78bfa', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
axes[0].fill_between(fpr, tpr, alpha=0.15, color='#a78bfa')
axes[0].plot([0, 1], [0, 1], '--', lw=1, color='#4a5568', label='Random baseline')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC-AUC Curve')
axes[0].legend(facecolor='#1a1a2e', edgecolor='#3a3a6e', labelcolor='#e2e8f0')

# Feature Importance
top15 = feat_imp.sort_values(ascending=True).tail(15)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top15)))
axes[1].barh(top15.index, top15.values, color=colors, edgecolor='none', height=0.7)
axes[1].set_title('Top 15 Feature Importances')
axes[1].set_xlabel('Importance Score')

plt.tight_layout(pad=2.0)
st.pyplot(fig)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">Kaggle Credit Card Fraud Detection</a>
    &nbsp;·&nbsp; Model: Random Forest with SMOTE
</div>
""", unsafe_allow_html=True)
