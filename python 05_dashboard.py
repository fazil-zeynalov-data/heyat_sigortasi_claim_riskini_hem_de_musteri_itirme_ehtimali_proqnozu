# 05_dashboard.py (Yenilənmiş)
import streamlit as st
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier

st.set_page_config(page_title="PASHA Həyat - 360° Müştəri Paneli", layout="wide", page_icon="🏥")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

@st.cache_data
def load_data():
    return pd.read_csv('features.csv')

@st.cache_resource
def train_models(df):
    # Risk Modeli
    drop_risk = ['customer_id', 'customer_since', 'n_claims', 'total_claim', 'avg_claim',
                 'claim_ratio', 'target_claim_flag', 'target_claim_amount', 'high_claim_history', 'target_churn']
    X_risk = df.drop(columns=[c for c in drop_risk if c in df.columns])
    y_risk = df.target_claim_flag
    model_risk = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    model_risk.fit(X_risk, y_risk)

    # Churn Modeli
    drop_churn = ['customer_id', 'customer_since', 'target_claim_flag', 'target_claim_amount', 'target_churn']
    X_churn = df.drop(columns=[c for c in drop_churn if c in df.columns])
    y_churn = df.target_churn
    model_churn = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
    model_churn.fit(X_churn, y_churn)

    return model_risk, X_risk.columns, model_churn, X_churn.columns

st.title("🏥 PASHA Həyat — 360° Müştəri Qiymətləndirmə Paneli")
st.markdown("Bu panel maşın öyrənməsi ilə həm **Claim Riskini**, həm də **Müştəri İtirmə (Churn) Ehtimalını** proqnozlaşdırır.")

df = load_data()
model_risk, cols_risk, model_churn, cols_churn = train_models(df)

st.sidebar.header("👤 Müştəri Profili")
age = st.sidebar.slider("Yaş", 18, 72, 35)
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
smoker = st.sidebar.selectbox("Siqaret çəkir?", ["Xeyr", "Bəli"])
chronic = st.sidebar.slider("Xroniki xəstəliklər", 0, 4, 0)
occupation = st.sidebar.slider("Peşə riski (0-3)", 0, 3, 1)
income = st.sidebar.number_input("Aylıq gəlir (AZN)", min_value=400, max_value=15000, value=1500)
n_policies = st.sidebar.slider("Məhsul sayı (Polislər)", 1, 3, 1)

# Risk Proqnozu
input_risk = pd.DataFrame(0, index=[0], columns=cols_risk)
input_risk['age'] = age; input_risk['bmi'] = bmi; input_risk['smoker'] = 1 if smoker=="Bəli" else 0
input_risk['chronic_conditions'] = chronic; input_risk['occupation_risk'] = occupation; input_risk['income_azn'] = income
prob_risk = model_risk.predict_proba(input_risk)[0, 1]

# Churn Proqnozu
input_churn = pd.DataFrame(0, index=[0], columns=cols_churn)
input_churn['age'] = age; input_churn['bmi'] = bmi; input_churn['n_policies'] = n_policies; input_churn['income_azn'] = income
prob_churn = model_churn.predict_proba(input_churn)[0, 1]

# Tarif
tariff_azn = 100 if prob_risk < 0.25 else (186 if prob_risk < 0.35 else (262 if prob_risk < 0.45 else (337 if prob_risk < 0.55 else (417 if prob_risk < 0.65 else 521))))

# Vizual Nəticələr
col1, col2, col3 = st.columns(3)
col1.metric("Claim Risk Ehtimalı", f"{prob_risk:.1%}")
col2.metric("Təklif olunan Tarif", f"{tariff_azn} AZN")
col3.metric("Müştəri İtirmə (Churn) Riski", f"{prob_churn:.1%}", delta_color="inverse")

st.markdown("---")
if prob_churn > 0.4:
    st.error("🚨 **DİQQƏT:** Bu müştərinin şirkətdən ayrılma ehtimalı yüksəkdir! Marketinq şöbəsi tərəfindən xüsusi endirim və ya loyallıq təklifi edilməlidir.")
else:
    st.success("✅ Bu müştərinin şirkətdə qalma ehtimalı yüksəkdir.")

st.subheader("📊 Risk Faktorlarının Təsiri (SHAP)")
try:
    st.image("shap_single_customer.png", caption="Claim Riskini təsir edən əsas faktorlar")
    st.image("shap_churn_global.png", caption="Churn (Ayrılma) səbəbləri")
except:
    st.info("SHAP qrafikləri hələ yaradılmayıb. Zəhmət olmasa əvvəlki skriptləri işə salın.")

st.markdown("---")
st.subheader("📋 Ümumi Tarif Cədvəli (Biznes Qaydası)")
st.table(pd.DataFrame({
    "Risk Sinifi": ["A+", "A", "B", "C", "D", "E"],
    "Claim Ehtimalı": ["< 25%", "25% - 35%", "35% - 45%", "45% - 55%", "55% - 65%", "> 65%"],
    "Təklif olunan Tarif (AZN)": [100, 186, 262, 337, 417, 521]
}))