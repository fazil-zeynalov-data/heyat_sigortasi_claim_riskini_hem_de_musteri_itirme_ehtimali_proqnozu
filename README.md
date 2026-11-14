# 🏥 PASHA Həyat - Fərdi Risk və Tarifləşdirmə (Data Science Layihəsi)

## 📌 Layihə Haqqında
Bu layihə sığorta şirkətləri üçün **maşın öyrənməsi (Machine Learning)** əsaslı fərdi risk qiymətləndirməsi və tarifləşdirmə sistemidir. Ənənəvi aktuar modellərini qabaqcıl süni intellekt alqoritmləri ilə əvəz edərək, həm müştəri məmnuniyyətini artırmaq, həm də şirkətin gəlirliliyini qorumaq məqsədi daşıyır.

## 🎯 Həll Etdiyi Biznes Problemləri
1. **Fərdi Tarifləşdirmə:** Müştərinin yaşı, BMI, siqaret vərdişi və xroniki xəstəliklərinə əsasən fərdi sığorta tarifi (A+ - E sinifləri) hesablanması.
2. **Müştəri İtirmə (Churn) Proqnozu:** Müştərinin şirkətdən ayrılma ehtimalının əvvəlcədən müəyyən edilməsi və marketinq şöbəsinin vaxtında tədbir görməsi.
3. **İzahlılıq (Explainability):** SHAP texnologiyası ilə modelin "niyə bu qərarı verdiyinin" şəffaf izahı (tənzimləyici orqanlar üçün).

## 🛠️ Texnologiyalar
* **Dil:** Python
* **Data Emalı:** Pandas, NumPy
* **Maşın Öyrənməsi:** XGBoost, Random Forest, Logistic Regression, Tweedie GLM (Statsmodels)
* **İzahlılıq:** SHAP
* **Vizualizasiya və Dashboard:** Streamlit, Matplotlib

## 📊 Nəticələr
* **Risk Modeli (AUC):** 0.76 (Sənaye standartlarına uyğun yüksək dəqiqlik)
* **Churn Modeli:** Müştəri itirmə riskini 80%+ dəqiqliklə proqnozlaşdırır.
* **Panel:** Real vaxtda işləyən, underwriting (sığorta qiymətləndirməsi) menecerləri üçün interaktiv Streamlit paneli.

## 🚀 Layihəni Necə İşə Salmaq Olar?
1. Repozitoriyanı klonlayın:
   ```bash
   git clone https://github.com/username/pasha-hayat-risk-pricing.git