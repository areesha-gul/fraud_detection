# =============================================================
#  Fraud Detection — Streamlit Demo App
#  Run: streamlit run app.py
#  Note: Train the model first by running fraud_detection.py
#        OR the model trains on startup (takes ~1-2 min)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Credit Card Fraud Detection")
st.markdown("**Credit Card Fraud Detection** | Random Forest + Logistic Regression")
st.divider()

# ------------------------------------------------------------
# LOAD & TRAIN (cached so it only runs once)
# ------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv('creditcard.csv')

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

    return rf, scaler, X_test, y_test, y_proba, fpr, tpr, roc_auc, feat_imp, df

# ------------------------------------------------------------
# SIDEBAR — MANUAL TRANSACTION INPUT
# ------------------------------------------------------------
st.sidebar.header("🧾 Test a Transaction")
st.sidebar.markdown("Adjust values to simulate a transaction:")

amount = st.sidebar.slider("Transaction Amount (USD)", 0.0, 5000.0, 120.0, 5.0)
time   = st.sidebar.slider("Time (seconds elapsed)", 0, 172800, 50000, 1000)

st.sidebar.markdown("---")
st.sidebar.markdown("**V-Features** (PCA-transformed, leave as 0 for typical transaction)")
v14 = st.sidebar.slider("V14 (most predictive)", -20.0, 10.0, 0.0, 0.1)
v17 = st.sidebar.slider("V17", -20.0, 10.0, 0.0, 0.1)
v12 = st.sidebar.slider("V12", -20.0, 10.0, 0.0, 0.1)
v10 = st.sidebar.slider("V10", -20.0, 10.0, 0.0, 0.1)

predict_btn = st.sidebar.button("🔎 Predict This Transaction", use_container_width=True)

# ------------------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------------------
with st.spinner("Training model on 284,807 transactions... (first load only, ~1 min)"):
    try:
        rf, scaler, X_test, y_test, y_proba, fpr, tpr, roc_auc, feat_imp, df = load_and_train()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

if not model_loaded:
    st.error("⚠️ `creditcard.csv` not found. Download it from Kaggle and place it in the same folder as app.py")
    st.markdown("[Download Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.stop()

# ------------------------------------------------------------
# PREDICTION PANEL
# ------------------------------------------------------------
if predict_btn:
    st.subheader("🔎 Prediction Result")

    # Build feature vector (30 features: V1-V28 + Amount_scaled + Time_scaled)
    sample = np.zeros(30)
    sample[13] = v14   # V14 is index 13 (0-indexed)
    sample[16] = v17
    sample[11] = v12
    sample[9]  = v10
    sample[28] = scaler.transform([[amount]])[0][0]     # Amount_scaled
    sample[29] = (time - 94813) / 47488                 # Time_scaled (approx)

    proba = rf.predict_proba([sample])[0][1]
    pred  = "🚨 FRAUD" if proba >= 0.5 else "✅ LEGITIMATE"
    color = "red" if proba >= 0.5 else "green"

    col1, col2, col3 = st.columns(3)
    col1.metric("Verdict", pred)
    col2.metric("Fraud Probability", f"{proba*100:.1f}%")
    col3.metric("Amount", f"${amount:,.2f}")

    st.progress(float(proba))

    if proba >= 0.5:
        st.error(f"⚠️ This transaction is flagged as **FRAUD** with {proba*100:.1f}% confidence.")
    else:
        st.success(f"✅ This transaction appears **LEGITIMATE** ({proba*100:.1f}% fraud probability).")
    st.divider()

# ------------------------------------------------------------
# METRICS OVERVIEW
# ------------------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC Score",  f"{roc_auc:.4f}",  "Random Forest")
col2.metric("Total Transactions", "284,807", "Kaggle dataset")
col3.metric("Fraud Cases",    "492",         "0.17% of all txns")
col4.metric("Technique", "SMOTE", "Class imbalance fix")

# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------
st.subheader("📈 ROC-AUC Curve")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(fpr, tpr, color='#E24B4A', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random baseline')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC-AUC Curve')
axes[0].legend()

top15 = feat_imp.sort_values(ascending=True).tail(15)
axes[1].barh(top15.index, top15.values, color='#378ADD')
axes[1].set_title('Top 15 Feature Importances')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.markdown(
    "Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · "
    "Model: Random Forest with SMOTE"
)
