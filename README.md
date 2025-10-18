# Credit-Risk-Prediction 📊

This project implements an end-to-end multi-class classification model to predict the credit risk category (P1, P2, P3, P4) for loan applicants. The final output is an interactive Streamlit web application designed for underwriting teams.

**This project was a rebuild focused on correcting foundational errors, particularly data leakage and improper handling of class imbalance.**

---

## 🎯 Business Problem

Lending institutions need to accurately assess applicant risk to minimize loan defaults (losses) while maximizing approvals for creditworthy customers (opportunities). This project aims to:

1.  **Automate Risk Assessment:** Provide an instant risk score (P1-P4) using only data available *at the time of application*.
2.  **Improve Consistency:** Reduce subjectivity in underwriting decisions.
3.  **Handle Data Challenges:** Build a robust model despite significant class imbalance and potential data quality issues.

---

## 🔑 Key Findings & Solutions (Project Principles)

This project strictly adhered to critical findings from a previous failed attempt:

1.  **Data Leakage (❗ #1 Priority):**
    * **Problem:** Target leakage (`Credit_Score`) and Feature leakage (numerous post-approval columns like payment history) were present.
    * **Solution:** All leaking features were identified using the data dictionary and **dropped immediately** after loading data, *before* any EDA or preprocessing. This ensures the model only uses pre-approval information.
2.  **Data Quality & Integrity:**
    * **Problem:** `-99999` used for nulls; high multicollinearity (e.g., `Total_TL` vs. `Tot_Active_TL + Tot_Closed_TL`).
    * **Solution:** `-99999` replaced with `np.nan` as the very first step. Multicollinearity was systematically addressed using **VIF analysis** *after* preprocessing, dropping composite features in favor of granular ones for better explainability.
3.  **Optimal Feature Encoding:**
    * **Problem:** Mixed feature types require specific encoding. `EDUCATION` has an inherent order.
    * **Solution:** `EDUCATION` was manually mapped using `OrdinalEncoder`. True nominal features (`GENDER`, `MARITALSTATUS`, etc.) were handled using `OneHotEncoder` with `drop='first'` to prevent the dummy variable trap.
4.  **Class Imbalance (Core Challenge):**
    * **Problem:** The dataset is dominated by the P2 (Low Risk) class, risking model bias.
    * **Solution:** The entire strategy revolved around this:
        * All `train_test_split` calls used `stratify=y`.
        * Modeling was done within an `imblearn.pipeline.Pipeline` that integrated **SMOTE** (Synthetic Minority Over-sampling Technique) correctly *only* on the training folds during cross-validation.
        * Evaluation focused on **`f1-macro`**, not just accuracy.

---

## 📁 Project Structure

/credit-risk-project/ |-- .streamlit/ | |-- config.toml # Streamlit theme configuration (dark mode) |-- app/ | |-- Home.py # Main Streamlit page (Dashboard) | |-- pages/ | | |-- 1_Batch_Prediction.py | | |-- 2_Single_Applicant_Prediction.py | | |-- 3_How_to_Use.py |-- data/ | |-- raw/ # Original applicant_data.csv, bureau_data.csv | |-- processed/ # Cleaned data (X_model_input.csv, y_target.csv) |-- docs/ # Business problem description, feature dictionary |-- models/ # Saved final model (tuned_model.joblib), preprocessor, encoders |-- notebooks/ # Jupyter notebooks for each project phase (01-05) |-- src/ # (Optional: Folder for reusable Python scripts/functions) |-- .gitignore |-- README.md # This file |-- requirements.txt # Project dependencies |-- venv/ # Virtual environment

---

## 🛠️ Workflow Summary

The project followed a standard data science lifecycle:

1.  **Phase 0: Scaffolding & Setup:** Defined project structure, virtual environment.
2.  **Phase 1: Data Ingestion & Cleaning:** Loaded data, immediately applied **Finding 1 (Leakage)** and null replacement (**Finding 2**).
3.  **Phase 2: EDA:** Analyzed features using statistical tests (Chi2, ANOVA), validated encoding needs (**Finding 3**), and planned missing data strategy. Pruned statistically insignificant features.
4.  **Phase 3: Feature Engineering & Selection:** Built the `ColumnTransformer` preprocessing pipeline (imputation, scaling, encoding). Systematically removed **multicollinearity** using VIF (**Finding 2**). Saved the final `preprocessor.joblib`.
5.  **Phase 4: Modeling Pipeline & Baseline:** Built an `imblearn.pipeline.Pipeline` integrating the preprocessor, **SMOTE (Finding 4)**, and a baseline `XGBClassifier`. Evaluated using stratified cross-validation (`f1-macro`). Saved `baseline_model.joblib`.
6.  **Phase 5: Hyperparameter Tuning:** Used **Optuna** to tune SMOTE and XGBoost parameters, focusing on improving the F1-score for the underperforming P3 class. Saved the improved `tuned_model.joblib`.
7.  **Phase 6: Deployment Preparation:** Built the multi-page **Streamlit application** for user interaction.

---

## ✨ Final Model & Performance

* **Algorithm:** Tuned `XGBClassifier` within an `imblearn` pipeline (Preprocessor -> SMOTE -> XGBoost).
* **Key Performance Metric:** `f1-macro` score.
    * Baseline Model (Phase 4): **0.60**
    * Tuned Model (Phase 5): **0.61** (Achieved goal of improving P3 class performance by +32%).
* **Final Features:** 38 features selected after rigorous leakage removal, VIF analysis, and EDA.

---

## 🚀 Streamlit Application

The final product is a multi-page Streamlit web app:

* **Home:** Dashboard explaining the project, defining risk classes, and showing key visualizations (data distribution, feature importance, income/age vs. risk).
* **Batch Prediction:** Allows users to upload a CSV of multiple applicants, get predictions, view results distribution (pie chart), and download predictions. Includes a downloadable sample CSV.
* **Single Applicant Prediction:** A user-friendly form (grouped sections, tooltips) to input data for one applicant and get an immediate prediction and recommendation.
* **How to Use:** Provides guidance on using the app and interpreting results.
* **Theme:** Configured for a dark theme.

---

## ⚙️ Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd credit-risk-project
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app/Home.py
    ```
    The app should open automatically in your browser.

---

## 💻 Key Technologies Used

* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (for SMOTE & pipeline)
* **Hyperparameter Tuning:** Optuna
* **Visualization:** Plotly Express, Matplotlib, Seaborn (in notebooks)
* **Web Application:** Streamlit
* **Utilities:** Joblib (for saving models/pipelines)