# 04_explainability.py
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Fayl yolunu təyin edirik
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("Data yüklənir...")
df = pd.read_csv('features.csv')

drop_cols = ['customer_id', 'customer_since', 'n_claims', 'total_claim', 'avg_claim',
             'claim_ratio', 'target_claim_flag', 'target_claim_amount',
             'high_claim_history']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y_flag = df.target_claim_flag

X_train, X_test, yf_train, yf_test = train_test_split(
    X, y_flag, test_size=0.2, random_state=42, stratify=y_flag
)

print("XGBoost modeli öyrədilir...")
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
                      scale_pos_weight=(y_flag==0).sum()/(y_flag==1).sum(), 
                      eval_metric='logloss', random_state=42)
model.fit(X_train, yf_train)

print("SHAP izahları hesablanır (bu bir neçə saniyə çəkə bilər)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Qlobal Feature Importance (Hansı faktorlar riskə daha çox təsir edir?)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_global_importance.png', dpi=150)
plt.close()
print("[OK] 'shap_global_importance.png' saxlanildi.")

# 2. Beeswarm Plot (Faktorların istiqaməti: artırır, yoxsa azaldır?)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150)
plt.close()
print("[OK] 'shap_beeswarm.png' saxlanildi.")

# 3. Bir müştəri üçün fərdi izah (Waterfall)
# Gəlin test setindən ilk müştərini götürək
plt.figure()
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0], 
    base_values=explainer.expected_value,
    data=X_test.iloc[0], 
    feature_names=X_test.columns.tolist()
), show=False)
plt.tight_layout()
plt.savefig('shap_single_customer.png', dpi=150)
plt.close()
print("[OK] 'shap_single_customer.png' saxlanildi.")

print("\n[OK] Butun SHAP qrafikleri ugurla yaradildi!")