import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.api as sm

# Fayl yollarını avtomatik tapan hissə
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print(f"Data qovluğu: {BASE_DIR}")
df = pd.read_csv('features.csv')

# Data sızmasının (leakage) qarşısını alırıq
drop_cols = ['customer_id', 'customer_since', 'n_claims', 'total_claim', 'avg_claim',
             'claim_ratio', 'target_claim_flag', 'target_claim_amount',
             'high_claim_history']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y_flag = df.target_claim_flag
y_amount = df.target_claim_amount

# Train/Test bölgüsü
X_train, X_test, yf_train, yf_test, ya_train, ya_test = train_test_split(
    X, y_flag, y_amount, test_size=0.2, random_state=42, stratify=y_flag
)

# Skalalaşdırma (LogReg üçün)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------- 1. KLASSİFİKASİYA ----------
print("\n" + "="*70)
print("KLASSİFİKASİYA — Claim ehtimalı")
print("="*70)

models = {
    'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=(y_flag==0).sum()/(y_flag==1).sum(), eval_metric='logloss', random_state=42)
}

best_clf = None
best_auc = 0

for name, m in models.items():
    m.fit(X_train_s if name=='LogReg' else X_train, yf_train)
    proba = m.predict_proba(X_test_s if name=='LogReg' else X_test)[:, 1]
    auc = roc_auc_score(yf_test, proba)
    pred = (proba > 0.25).astype(int)
    print(f"{name:15s} | AUC: {auc:.4f} | Prec: {precision_score(yf_test,pred):.3f} | Rec: {recall_score(yf_test,pred):.3f} | F1: {f1_score(yf_test,pred):.3f}")
    if auc > best_auc:
        best_auc = auc
        best_clf = (name, m)

# ---------- 2. REGRESSİYA ----------
print("\n" + "="*70)
print("REGRESSİYA — Claim məbləği")
print("="*70)

# Tweedie GLM
glm = sm.GLM(ya_train, sm.add_constant(X_train_s), family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())).fit()
pred_glm = glm.predict(sm.add_constant(X_test_s))

# XGBoost Tweedie
xgb_reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, objective='reg:tweedie', tweedie_variance_power=1.5, random_state=42)
xgb_reg.fit(X_train, ya_train)
pred_xgb = xgb_reg.predict(X_test)

# RandomForest
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, ya_train)
pred_rf = rf_reg.predict(X_test)

for name, pred in [('Tweedie GLM', pred_glm), ('XGBoost-Tweedie', pred_xgb), ('RandomForest', pred_rf)]:
    print(f"{name:18s} | MAE: {mean_absolute_error(ya_test,pred):8.2f}")

# ---------- 3. TARİF CƏDVƏLİ ----------
df_test = X_test.copy()
df_test['prob_claim'] = best_clf[1].predict_proba(X_test_s if best_clf[0]=='LogReg' else X_test)[:, 1]
df_test['tariff_class'] = pd.qcut(df_test.prob_claim, 6, labels=['A+','A','B','C','D','E'])

tariff = df_test.groupby('tariff_class', observed=True).agg(
    musteri=('prob_claim', 'size'),
    orta_risk=('prob_claim', 'mean')
).round(4)

tariff['tarif_AZN'] = (100 * (tariff.orta_risk / tariff.orta_risk.min())).round(0)

print("\n" + "="*70)
print("TARİF CƏDVƏLİ (Baza = 100 AZN)")
print("="*70)
print(tariff)
print("\n[OK] Proses ugurla tamamlandi!")