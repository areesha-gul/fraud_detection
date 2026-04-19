# =============================================================
#  Fraud Detection ML Project
#  Dataset: Kaggle Credit Card Fraud Detection
#  Models:  Logistic Regression + Random Forest
#  Author:  [Your Name]
# =============================================================

# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the same folder as this script

print("=" * 60)
print("  FRAUD DETECTION ML PROJECT")
print("=" * 60)

print("\n[1/6] Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"  Shape         : {df.shape}")
print(f"  Total txns    : {len(df):,}")
print(f"  Fraud cases   : {df['Class'].sum():,}")
print(f"  Fraud rate    : {df['Class'].mean()*100:.4f}%")
print(f"  Missing values: {df.isnull().sum().sum()}")

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
print("\n[2/6] Running EDA and saving charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fraud Detection — Exploratory Data Analysis', fontsize=15, fontweight='bold')

# 3a. Class distribution
class_counts = df['Class'].value_counts()
axes[0, 0].bar(['Legitimate', 'Fraud'], class_counts.values,
                color=['#378ADD', '#E24B4A'], edgecolor='white', linewidth=0.5)
axes[0, 0].set_title('Class Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Number of Transactions')
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=10)

# 3b. Transaction amount distribution
axes[0, 1].hist(df[df['Class'] == 0]['Amount'], bins=60, alpha=0.7,
                color='#378ADD', label='Legitimate', density=True)
axes[0, 1].hist(df[df['Class'] == 1]['Amount'], bins=60, alpha=0.7,
                color='#E24B4A', label='Fraud', density=True)
axes[0, 1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Amount (USD)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_xlim(0, 1000)
axes[0, 1].legend()

# 3c. Transactions over time
axes[1, 0].scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'],
                   alpha=0.05, s=1, color='#378ADD', label='Legitimate')
axes[1, 0].scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'],
                   alpha=0.4, s=5, color='#E24B4A', label='Fraud')
axes[1, 0].set_title('Transactions Over Time', fontweight='bold')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Amount (USD)')
axes[1, 0].legend()

# 3d. Top correlated features with fraud
correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)
top10 = correlations.head(10)
colors = ['#E24B4A' if df.corr()['Class'][f] < 0 else '#378ADD' for f in top10.index]
axes[1, 1].barh(top10.index[::-1], top10.values[::-1], color=colors[::-1])
axes[1, 1].set_title('Top 10 Features Correlated with Fraud', fontweight='bold')
axes[1, 1].set_xlabel('Absolute Correlation')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: eda_plots.png")

# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------
print("\n[3/6] Preprocessing data...")

# Scale Amount and Time (V1–V28 are already scaled by PCA)
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])

# Drop original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size : {len(X_train):,}")
print(f"  Test size  : {len(X_test):,}")

# Apply SMOTE only to training set
print("  Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE — Fraud: {y_train_sm.sum():,} | Legit: {(y_train_sm==0).sum():,}")

# ------------------------------------------------------------
# 5. TRAIN MODELS
# ------------------------------------------------------------
print("\n[4/6] Training models...")

# --- Logistic Regression ---
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sm, y_train_sm)
y_pred_lr    = lr.predict(X_test)
y_proba_lr   = lr.predict_proba(X_test)[:, 1]
print("  Done.")

# --- Random Forest ---
print("  Training Random Forest (this may take 1–2 min)...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf    = rf.predict(X_test)
y_proba_rf   = rf.predict_proba(X_test)[:, 1]
print("  Done.")

# ------------------------------------------------------------
# 6. EVALUATE MODELS
# ------------------------------------------------------------
print("\n[5/6] Evaluating models...")

print("\n  --- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr, target_names=['Legit', 'Fraud']))

print("  --- Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=['Legit', 'Fraud']))

# ------------------------------------------------------------
# 7. VISUALIZE RESULTS
# ------------------------------------------------------------
print("[6/6] Generating result charts...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Fraud Detection — Model Evaluation Results', fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 7a. Confusion Matrix — Logistic Regression
ax1 = fig.add_subplot(gs[0, 0])
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(cm_lr, display_labels=['Legit', 'Fraud'])
disp_lr.plot(ax=ax1, colorbar=False, cmap='Blues')
ax1.set_title('Confusion Matrix\nLogistic Regression', fontweight='bold')

# 7b. Confusion Matrix — Random Forest
ax2 = fig.add_subplot(gs[0, 1])
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=['Legit', 'Fraud'])
disp_rf.plot(ax=ax2, colorbar=False, cmap='Oranges')
ax2.set_title('Confusion Matrix\nRandom Forest', fontweight='bold')

# 7c. ROC-AUC Curve (both models)
ax3 = fig.add_subplot(gs[0, 2])
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
ax3.plot(fpr_lr, tpr_lr, color='#378ADD', lw=2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax3.plot(fpr_rf, tpr_rf, color='#E24B4A', lw=2, label=f'Random Forest       (AUC = {auc_rf:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random baseline')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.02])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC-AUC Curve\nBoth Models', fontweight='bold')
ax3.legend(fontsize=9)

# 7d. Precision-Recall Curve
ax4 = fig.add_subplot(gs[1, 0])
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
ap_lr = average_precision_score(y_test, y_proba_lr)
ap_rf = average_precision_score(y_test, y_proba_rf)
ax4.plot(rec_lr, prec_lr, color='#378ADD', lw=2, label=f'LR  (AP = {ap_lr:.3f})')
ax4.plot(rec_rf, prec_rf, color='#E24B4A', lw=2, label=f'RF  (AP = {ap_rf:.3f})')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve\n(Better for imbalanced data)', fontweight='bold')
ax4.legend(fontsize=9)

# 7e. Feature Importance — Random Forest (top 15)
ax5 = fig.add_subplot(gs[1, 1:])
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top15 = feat_imp.sort_values(ascending=True).tail(15)
colors_fi = ['#E24B4A' if 'V' in f else '#378ADD' for f in top15.index]
ax5.barh(top15.index, top15.values, color=colors_fi, edgecolor='white')
ax5.set_title('Random Forest — Top 15 Feature Importances', fontweight='bold')
ax5.set_xlabel('Importance Score')

plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: model_results.png")

# ------------------------------------------------------------
# 8. SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Logistic Regression  AUC : {auc_lr:.4f}")
print(f"  Random Forest        AUC : {auc_rf:.4f}")
print(f"\n  Random Forest Confusion Matrix:")
print(f"    True Negatives  (legit correctly rejected) : {cm_rf[0,0]:,}")
print(f"    False Positives (false alarms)             : {cm_rf[0,1]:,}")
print(f"    False Negatives (missed fraud!)            : {cm_rf[1,0]:,}")
print(f"    True Positives  (fraud caught)             : {cm_rf[1,1]:,}")
recall_rf = cm_rf[1,1] / (cm_rf[1,0] + cm_rf[1,1])
print(f"\n  Fraud Recall (Random Forest) : {recall_rf*100:.1f}%")
print(f"  i.e. the model catches {recall_rf*100:.1f}% of all actual fraud cases")
print("\n  Charts saved:")
print("    eda_plots.png    — EDA visualisations")
print("    model_results.png — Confusion matrices, ROC, PR, Feature importance")
print("\n  Done! Add these results + GitHub link to your CV.")
print("=" * 60)
