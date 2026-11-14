import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 50000  # müştəri sayı

# ============ 1. MÜŞTƏRİ CƏDVƏLİ ============
customers = pd.DataFrame({
    'customer_id': [f'C{i:06d}' for i in range(N)],
    'age': np.random.randint(18, 72, N),
    'gender': np.random.choice(['M', 'F'], N, p=[0.52, 0.48]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], N, p=[0.35, 0.55, 0.10]),
    'education': np.random.choice(['Secondary', 'Bachelor', 'Master', 'PhD'], N, p=[0.35, 0.45, 0.17, 0.03]),
    'occupation_risk': np.random.choice([0, 1, 2, 3], N, p=[0.45, 0.30, 0.18, 0.07]),  # 0=ofis, 3=yüksək risk
    'income_azn': np.clip(np.random.lognormal(9.5, 0.6, N), 400, 15000).round(0),
    'bmi': np.clip(np.random.normal(26.5, 4.5, N), 15, 45),
    'smoker': np.random.choice([0, 1], N, p=[0.72, 0.28]),
    'alcohol': np.random.choice([0, 1, 2], N, p=[0.60, 0.30, 0.10]),  # 0=yox, 1=orta, 2=çox
    'chronic_conditions': np.clip(np.random.poisson(0.45, N), 0, 4),
    'family_history': np.random.choice([0, 1], N, p=[0.70, 0.30]),  # irsi xəstəlik
    'region': np.random.choice(['Baku', 'Ganja', 'Sumgait', 'Other_City', 'Rural'], N,
                                p=[0.48, 0.10, 0.08, 0.14, 0.20]),
    'customer_since': [datetime(2020,1,1) + timedelta(days=int(d))
                       for d in np.random.randint(0, 1800, N)],
    'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], N,
                                      p=[0.5, 0.3, 0.15, 0.05]),
})

# ============ 2. POLİS CƏDVƏLİ (hər müştəri 1-3 məhsul ala bilər) ============
product_catalog = {
    'Life_Term':      {'base_premium': 250,  'coverage': 100000, 'risk_weight': 0.8},
    'Life_Whole':     {'base_premium': 800,  'coverage': 250000, 'risk_weight': 1.0},
    'Health_Basic':   {'base_premium': 400,  'coverage': 20000,  'risk_weight': 1.2},
    'Health_Premium': {'base_premium': 1200, 'coverage': 80000,  'risk_weight': 1.5},
    'Critical_Illness': {'base_premium': 350,'coverage': 50000,  'risk_weight': 1.3},
    'Accident':       {'base_premium': 150,  'coverage': 30000,  'risk_weight': 0.9},
    'Travel':         {'base_premium': 100,  'coverage': 15000,  'risk_weight': 0.5},
}

policies_list = []
for cid in customers.customer_id:
    n_products = np.random.choice([1, 2, 3], p=[0.55, 0.32, 0.13])
    chosen = np.random.choice(list(product_catalog.keys()), n_products, replace=False)
    for prod in chosen:
        p_start = datetime(2020,1,1) + timedelta(days=int(np.random.randint(0, 1500)))
        policies_list.append({
            'policy_id': f'P{len(policies_list):07d}',
            'customer_id': cid,
            'product': prod,
            'base_premium': product_catalog[prod]['base_premium'],
            'coverage': product_catalog[prod]['coverage'],
            'start_date': p_start,
            'duration_months': np.random.choice([12, 24, 36], p=[0.7, 0.2, 0.1]),
            'payment_freq': np.random.choice(['Monthly', 'Quarterly', 'Annual'], p=[0.55, 0.25, 0.20]),
        })
policies = pd.DataFrame(policies_list)

# ============ 3. CLAIM CƏDVƏLİ (aktuar risk modelinə əsasən) ============
# Hər müştəri üçün gizli risk skoru
cust = customers.copy()
risk_score = (
    0.035 * (cust.age - 35) +
    0.45  * cust.occupation_risk +
    0.09  * (cust.bmi - 25) +
    1.0   * cust.smoker +
    0.35  * cust.alcohol +
    0.55  * cust.chronic_conditions +
    0.40  * cust.family_history +
    0.15  * (cust.region == 'Baku').astype(int) +
    np.random.normal(0, 0.5, len(cust))
)
cust['prob_claim_base'] = 1 / (1 + np.exp(-(risk_score - 3.0)))

# Hər polis üçün claim ehtimalı = base * product risk_weight
claims_list = []
policy_risk = policies.merge(
    cust[['customer_id', 'prob_claim_base', 'chronic_conditions', 'smoker']],
    on='customer_id'
)
policy_risk['risk_weight'] = policy_risk['product'].map(
    {k: v['risk_weight'] for k, v in product_catalog.items()}
)
policy_risk['prob_claim'] = np.clip(
    policy_risk['prob_claim_base'] * policy_risk['risk_weight'], 0.001, 0.6
)

for _, row in policy_risk.iterrows():
    n_claims = np.random.poisson(row['prob_claim'] * 2)  # polis dövründə claim sayı
    for _ in range(n_claims):
        # Claim məbləği: gamma paylanması, xroniki xəstəlikdən asılı
        severity = np.random.gamma(
            shape=2.0,
            scale=400 * (1 + 0.3 * row['chronic_conditions']) * (1 + 0.4 * row['smoker'])
        )
        claims_list.append({
            'claim_id': f'CL{len(claims_list):07d}',
            'policy_id': row['policy_id'],
            'customer_id': row['customer_id'],
            'product': row['product'],
            'claim_date': row['start_date'] + timedelta(days=int(np.random.randint(30, 700))),
            'claim_amount': round(severity, 2),
            'claim_type': np.random.choice(['Hospital', 'Surgery', 'Diagnosis', 'Accident', 'Death'], 
                                            p=[0.35, 0.20, 0.25, 0.15, 0.05]),
            'status': np.random.choice(['Approved', 'Rejected', 'Pending'], p=[0.82, 0.10, 0.08]),
        })
claims = pd.DataFrame(claims_list)

# ============ Yadda saxla ============
customers.to_csv('customers.csv', index=False)
policies.to_csv('policies.csv', index=False)
claims.to_csv('claims.csv', index=False)

print(f"✓ Müştəri: {len(customers):,}")
print(f"✓ Polis:   {len(policies):,}")
print(f"✓ Claim:   {len(claims):,}")
print(f"✓ Claim rate (polis üzrə): {claims.policy_id.nunique()/len(policies):.2%}")
print(f"✓ Ortalama claim: {claims.claim_amount.mean():.2f} AZN")