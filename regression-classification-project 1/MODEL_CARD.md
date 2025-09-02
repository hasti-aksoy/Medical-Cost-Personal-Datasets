# MODEL CARD — Medical Cost Prediction (Linear Family)

**Artifact:** `models/final_linear_pipeline.joblib`  
**Version:** v1.0  
**Date:** <2025-09-02>  
**Author/Owner:** <Hati_Aksoy>  
**Contact:** <akhasti14@gmail.com>

---

## 1. Model Details
- **Type:** Linear regression (selected via Nested CV)
- **Framework:** scikit-learn
- **Pipeline:** `ColumnTransformer(StandardScaler + OneHotEncoder)` → **LinearRegression**
- **Feature engineering:** domain-driven interactions (`smoker×age`, `smoker×bmi`, `age×bmi`)
- **Target:** `charges` (USD).

## 2. Intended Use
- **Goal:** Predict individual medical insurance **charges** to enable educational analysis/demos of regression modeling, not for pricing in production.
- **Users:** Students, researchers, interviewers assessing ML skills.
- **Out-of-scope:** Clinical decisions, pricing policies, underwriting, or any real-world financial decision-making impacting individuals.

## 3. Data
- **Source:** Kaggle “Medical Cost Personal Datasets” (commonly known as *insurance* dataset).
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- **Population:** U.S. individuals; limited sample size.
- **Splits:** 80/20 train/test; KFold CV; **Nested CV** for model selection.

## 4. Preprocessing
- Numeric → StandardScaler; Categorical → OneHotEncoder(drop='first').
- Interactions added: `smoker×age`, `smoker×bmi`, `age×bmi`.
- No global PolynomialFeatures (to avoid feature explosion).

## 5. Metrics (Test Set)
- **RMSE:** 4567.93
- **MAE:** 2760.61
- **R²:** 0.866
- **MAPE%:** 29.37%
- **RMSLE:** 0.3945

## 6. Evaluation Procedures
- **Cross-validation:** 5-fold with shuffle (seed=42); **Nested CV** for unbiased model selection.
- **Diagnostics:** residual plots, normality (D’Agostino), heteroskedasticity (Breusch–Pagan).
- **Subgroup analysis:** error tables by `smoker`, `sex`, `region`, `age` bins.

## 7. Ethical Considerations
- `smoker` encodes behavioral/health status; the model may **systematically under/over-predict** for certain groups.
- The dataset is not representative of all populations; risk of **distribution shift** in other geographies.
- The model is **not** designed for consequential decisions (pricing/coverage/eligibility).

## 8. Caveats & Recommendations
- Linear models may miss complex interactions; consider domain-validated features if used beyond education.
- Evaluate on original scale and report multiple metrics (RMSE + MAE + RMSLE).
- Monitor subgroup performance; reconsider feature design if large disparities are observed.
- Keep the input schema consistent (names/types) with training.

## 9. How to Use
```python
import joblib, pandas as pd
pipe = joblib.load("models/final_linear_pipeline.joblib")
sample = pd.DataFrame([{  # doubled braces to render literal dict
    "age": 40, "sex": "female", "bmi": 29.5, "children": 2,
    "smoker": "no", "region": "southeast"
}])
float(pipe.predict(sample))
```

## 10. Versioning
- **v1.0:** Initial release with Linear/Ridge/Lasso/ElasticNet comparison; **Linear** selected via Nested CV.
