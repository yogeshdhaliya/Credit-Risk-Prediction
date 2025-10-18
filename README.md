# Credit Risk Prediction Model & Streamlit App 📊

![Credit Risk Prediction](https://img.shields.io/badge/Status-Complete-green) ![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

This project implements an **end-to-end multi-class classification model** to predict the credit risk category (**P1, P2, P3, P4**) for loan applicants. The final deliverable is an interactive **Streamlit web application** designed for underwriting teams to automate and streamline risk assessment.

---

## 🎯 Business Problem

Lending institutions must accurately assess applicant risk to minimize defaults (financial losses) while approving creditworthy customers (missed opportunities). Manual underwriting is slow, inconsistent, and prone to bias.

**This Project Solves It By:**
- **Automating Assessment:** Instant P1-P4 risk scoring using *pre-approval* data only.
- **Ensuring Consistency:** Data-driven decisions reduce subjectivity.
- **Handling Challenges:** Robust modeling despite class imbalance and data quality issues.

---

## 🔑 Challenges & Solutions

The project tackled core data and modeling challenges head-on:

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Data Leakage** (Top Priority) | Target leakage (`Credit_Score`) and 26+ post-approval features (e.g., payment history). | Dropped all leaking columns *immediately* after loading, using data dictionary analysis *before* EDA/preprocessing. Ensures model uses only valid pre-approval data. |
| **Data Quality & Multicollinearity** | `-99999` null placeholders; redundant totals (e.g., `Total_TL` = `Active_TL + Closed_TL`). | Replaced `-99999` with `np.nan` upfront. Used **VIF analysis** post-preprocessing to drop composites; kept granular features (e.g., `Auto_TL`) for explainability. |
| **Feature Encoding** | Mixed categorical types; ordinal `EDUCATION`; dummy trap risk. | Custom `OrdinalEncoder` for `EDUCATION`. `OneHotEncoder(drop='first')` for nominals (e.g., `GENDER`). |
| **Class Imbalance** | ~63% P2 dominance; minorities (P1, P3, P4) at risk of being ignored. | `stratify=y` in splits; **SMOTE** in `imblearn.pipeline.Pipeline` (applied *only* to training folds in CV); evaluated via `f1-macro`. |

---

## 📁 Project Structure

```
credit-risk-project/
├── .streamlit/
│   └── config.toml                             # Dark theme configuration
├── app/
│   ├── Home.py                                 # Dashboard with insights & navigation
│   └── pages/
│       ├── 1_Batch_Prediction.py               # CSV upload & batch predictions
│       ├── 2_Single_Applicant_Prediction.py    # Manual form for one applicant
│       └── 3_How_to_Use.py                     # User guide & interpretations
├── data/
│   ├── raw/                                    # Original CSVs (applicant_data.csv, bureau_data.csv)
│   └── processed/                              # Cleaned data (X_model_input.csv, y_target.csv)
├── docs/                                       # Business memos, feature dictionary
├── models/                                     # Pipelines (tuned_model.joblib, preprocessor.joblib)
├── notebooks/                                  # Phase-wise Jupyter notebooks (01_EDA.ipynb, etc.)
├── src/                                        # Reusable scripts (e.g., data_loader.py)
├── .gitignore
├── README.md                                   # This file
├── requirements.txt                            # Dependencies
└── venv/                                       # Virtual environment
```

---

## 🛠️ Workflow Summary

The project follows a structured ML lifecycle across 7 phases:

1. **Phase 0: Setup** – Project scaffolding, venv, requirements.
2. **Phase 1: Data Loading & Cleaning** – Merge CSVs, handle leakage and nulls.
3. **Phase 2: EDA** – Stats tests (Chi2, ANOVA), visualizations; prune insignificant features.
4. **Phase 3: Feature Engineering** – Build `ColumnTransformer` (impute, scale, encode); VIF-based selection → 38 features.
5. **Phase 4: Baseline Modeling** – `imblearn.Pipeline` (Preprocess → SMOTE → XGBoost); Stratified CV with `f1-macro`.
6. **Phase 5: Tuning** – **Optuna** (50 trials) for SMOTE/XGBoost params; focus on P3 improvement.
7. **Phase 6: Deployment** – Multi-page Streamlit app with UI/UX enhancements.

---

## ✨ Final Model & Performance

* **Algorithm:** Tuned `XGBClassifier` within an `imblearn.pipeline.Pipeline` (Preprocessor → SMOTE → XGBoost).
* **Key Performance Metric:** `f1-macro` score (chosen for robustness against class imbalance).
* **Final Features:** 38 features selected after rigorous leakage removal, VIF analysis, and EDA (examples include enquiry history, income, account age).
* **Performance:**
    * **Baseline Model (Phase 4):**
        * Test `f1-macro`: **0.60**
        * Test F1-Scores per Class: P1: 0.67, P2: 0.83, **P3: 0.25**, P4: 0.63
    * **Tuned Model (Phase 5):**
        * Test `f1-macro`: **0.61** (Overall improvement)
        * Test F1-Scores per Class: P1: 0.67, P2: 0.81, **P3: 0.33** (+32% vs Baseline), P4: 0.63
* **Stability:** Low standard deviation in cross-validation scores and a close match between CV and final test scores indicate a reliable and non-overfit model.

---

## 🚀 Streamlit Application

A multi-page app for seamless predictions:

- **Home:** Project overview, risk class definitions, key visuals (pie chart for distribution, bar for feature importance, box plots for income/age vs. risk).
- **Batch Prediction:** Upload CSV → Predictions → Pie chart of results + table → Download CSV. Includes 50-row sample CSV.
- **Single Prediction:** Collapsible form (grouped sections, tooltips, defaults) → Instant P1-P4 result + recommendation.
- **How to Use:** Step-by-step guide, interpretations (e.g., "Low income + high enquiries → P4").

**Theme:** Dark mode via `.streamlit/config.toml`.

**Run:** `streamlit run app/Home.py`.

---

## ⚙️ Setup & Usage

1. **Clone Repo:**
   ```bash
   git clone <your-repo-url>
   cd credit-risk-project
   ```

2. **Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch App:**
   ```bash
   streamlit run app/Home.py
   ```

---

## 💻 Key Technologies

- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Tuning:** Optuna
- **Viz:** Plotly Express, Matplotlib, Seaborn
- **App:** Streamlit
- **Utils:** Joblib (model saving)

---

## 📚 Resources

- **Notebooks:** `./notebooks/` for phase-by-phase code.
- **Models:** `./models/` for pipelines.
- **Docs:** `./docs/` for memos and feature glossary.

## 🤝 Contributing

Fork, PR, or raise issues. Follow PEP8.

## 📄 License

MIT License – See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for data-driven lending. Questions? Reach out!*