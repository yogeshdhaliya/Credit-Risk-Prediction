import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np # Needed for AGE/Income plot

# --- 1. Configuration ---
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="📊",
    layout="wide"
)

# --- 2. Define File Paths ---
PROJECT_ROOT = "/Users/yogeshdhaliya/Desktop/DS Learning/11. Projects/Credit-Risk-Prediction"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tuned_model.joblib")
Y_TARGET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "y_target.csv")
X_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "X_model_input.csv")

# --- 3. Load Assets (Cached) ---
# Use decorators AFTER import st
@st.cache_resource
def load_model():
    """Loads the tuned model pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Tuned model file not found at {MODEL_PATH}")
        return None

@st.cache_data
def load_data():
    """Loads target and features for visualization."""
    try:
        y = pd.read_csv(Y_TARGET_PATH, index_col='PROSPECTID')
        x_viz = pd.read_csv(X_INPUT_PATH, index_col='PROSPECTID', usecols=['PROSPECTID', 'AGE', 'NETMONTHLYINCOME'])
        viz_df = pd.concat([y, x_viz], axis=1)
        return y, viz_df
    except FileNotFoundError:
        st.error(f"Error: Data files not found. Check paths: {Y_TARGET_PATH}, {X_INPUT_PATH}")
        return None, None

model = load_model()
y, viz_df = load_data()

# --- 4. Page Content ---

st.title("📊 Credit Risk Prediction Dashboard")

st.markdown("""
Welcome! This application uses a machine learning model (**XGBoost**) to predict the credit risk category for loan applicants.
It helps underwriters make faster, data-driven decisions.
""")

# --- Section: Get Started Buttons (MOVED TO TOP) ---
st.subheader("🚀 Make a Prediction")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_Batch_Prediction.py", label="Predict for Multiple Applicants", icon="📂", use_container_width=True)
    st.caption("Upload a CSV file to get predictions for a batch of applicants.")
with col2:
    st.page_link("pages/2_Single_Applicant_Prediction.py", label="Predict for One Applicant", icon="👤", use_container_width=True)
    st.caption("Enter details manually to get a prediction for a single applicant.")
st.divider() # Add divider after buttons
# --- End Get Started Buttons ---


# --- Section: Business Problem ---
with st.expander("🎯 The Business Problem Solved", expanded=False): # Start collapsed
    st.markdown("""
    Lending institutions face the challenge of accurately assessing the risk associated with loan applicants. A wrong decision can lead to financial losses (default) or missed opportunities (rejecting a good applicant).

    **This project addresses this by:**
    * **Automating Risk Assessment:** Providing instant risk categorization (P1-P4) based on applicant data available *at the time of application*.
    * **Improving Consistency:** Reducing human bias and ensuring consistent application of risk rules.
    * **Handling Imbalance:** Explicitly addressing the challenge that good applicants (P2) vastly outnumber risky ones, ensuring the model learns to identify rare but critical high-risk (P4) and moderate-risk (P3) cases using **SMOTE**.
    * **Ensuring Validity:** Rigorously preventing **data leakage** (our #1 priority) and multicollinearity to build a trustworthy model based only on pre-approval information.
    """)

# --- Section: Risk Classes Defined ---
with st.expander("🚦 Understanding the Risk Categories (P1-P4)", expanded=False): # Start collapsed
    st.markdown("""
    The model assigns one of four risk categories:

    * <span style="color: #2ECC71;">**P1 (Lowest Risk):**</span> ✅ Premium applicants with the strongest profiles. Very high likelihood of repayment.
    * <span style="color: #3498DB;">**P2 (Low Risk):**</span> ✔️ Good, standard applicants meeting typical criteria. Reliable borrowers.
    * <span style="color: #F39C12;">**P3 (Moderate Risk):**</span> ⚠️ Subprime applicants showing some potential risk factors (e.g., higher enquiries, lower income stability). Require careful review.
    * <span style="color: #E74C3C;">**P4 (Highest Risk):**</span> ❌ High-risk applicants with multiple strong indicators suggesting potential default. Recommend rejection.
    """, unsafe_allow_html=True)


st.divider() # Add another divider before charts

# --- Section: Visual Insights ---
if model is None or y is None or viz_df is None:
    st.error("Application could not load necessary assets. Visualizations unavailable.")
else:
    st.header("🔍 Project Insights & Visualizations")
    # (Keep the visualization code exactly as it was)
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.subheader("1. Target Class Distribution")
        st.write("Shows the original class imbalance **(Finding 4)**. Our model uses **SMOTE** to handle this.")
        target_counts = y['Approved_Flag'].value_counts().reset_index()
        target_counts.columns = ['Risk Category', 'Count']
        fig_pie = px.pie(
            target_counts, names='Risk Category', values='Count',
            title='Original Data: Risk Category Distribution', hole=0.3,
            color_discrete_map={'P1':'#2ECC71', 'P2':'#3498DB', 'P3':'#F39C12', 'P4':'#E74C3C'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("3. Income vs. Risk")
        st.write("Illustrates how Net Monthly Income (log-scaled) varies across risk categories. Lower income is generally associated with higher risk (P3/P4).")
        viz_df['LogIncome'] = np.log1p(viz_df['NETMONTHLYINCOME'])
        fig_income = px.box(
            viz_df.sort_values('Approved_Flag'), x='Approved_Flag', y='LogIncome', color='Approved_Flag',
            title='Log(Net Monthly Income) Distribution by Risk Category',
            labels={'Approved_Flag': 'Risk Category', 'LogIncome': 'Log(Net Monthly Income + 1)'},
            color_discrete_map={'P1':'#2ECC71', 'P2':'#3498DB', 'P3':'#F39C12', 'P4':'#E74C3C'}
        )
        st.plotly_chart(fig_income, use_container_width=True)
    with viz_col2:
        st.subheader("2. Model Feature Importance")
        st.write("Shows the Top 15 features our **XGBoost model** uses to make decisions **(Explainability)**.")
        try:
            preprocessor = model.named_steps['preprocessor']
            xgb_model = model.named_steps['model']
            feature_names = preprocessor.get_feature_names_out()
            importance_df = pd.DataFrame({'Feature': feature_names,'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False).head(15)
            fig_bar = px.bar(
                importance_df.sort_values(by='Importance', ascending=True), x='Importance', y='Feature', orientation='h',
                title='Top 15 Most Important Features in Tuned Model'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate feature importance plot: {e}")

        st.subheader("4. Age vs. Risk")
        st.write("Shows how the age distribution differs across risk categories.")
        fig_age = px.box(
            viz_df.sort_values('Approved_Flag'), x='Approved_Flag', y='AGE', color='Approved_Flag',
            title='Age Distribution by Risk Category',
            labels={'Approved_Flag': 'Risk Category', 'AGE': 'Applicant Age'},
            color_discrete_map={'P1':'#2ECC71', 'P2':'#3498DB', 'P3':'#F39C12', 'P4':'#E74C3C'}
        )
        st.plotly_chart(fig_age, use_container_width=True)