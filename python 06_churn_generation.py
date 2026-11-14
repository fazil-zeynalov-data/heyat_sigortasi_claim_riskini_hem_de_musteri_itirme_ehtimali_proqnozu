# 06_churn_generation.py
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("Müştəri itirmə (Churn) datasi generasiya olunur...")
df = pd.read_csv('features.csv')

np.random.seed(42)
N = len(df)

# Churn ehtimalını təyin edən gizli riyazi düstur (Biznes məntiqi)
# Müştərilər adətən yüksək tarif, çox şikayət (claim) və aşağı sadiqlik olduqda ayrılırlar.
churn_logit = (
    -1.5 
    - 0.8 * (df['loyalty_tier_Gold'] + df['loyalty_tier_Platinum']) # Sadiq müştərilər az ayrılır
    + 0.6 * (df['n_policies'] == 1).astype(int) # Tək məhsulu olanlar asan ayrılır
    + 1.2 * df['premium_to_income'] # Tarif gəlirə nisbətən çox bahadırsa
    + 0.5 * df['high_claim_history'] # Çox claim edənlər (problemli müştərilər) və ya rədd edilənlər
    + 0.3 * (df['age'] < 25).astype(int) # Gənclər daha mobildir
    + np.random.normal(0, 0.5, N) # Təsadüfilik
)

# Ehtimalı 0-1 arasına salırıq
prob_churn = 1 / (1 + np.exp(-churn_logit))

# Həqiqi churn statusunu təyin edirik (1 = Ayrıldı, 0 = Qaldı)
df['target_churn'] = np.random.binomial(1, prob_churn)

# Faylı yeniləyirik
df.to_csv('features.csv', index=False)
print(f"[OK] Churn datasi elave edildi. Umumi Churn rate: {df['target_churn'].mean():.2%}")