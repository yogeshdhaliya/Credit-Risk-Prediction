import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. Configuration and Asset Loading ---
st.set_page_config(
    page_title="Credit Risk | Single Prediction",
    page_icon="👤",
    layout="wide"
)

# --- 2. Define File Paths ---
PROJECT_ROOT = "/Users/yogeshdhaliya/Desktop/DS Learning/11. Projects/Credit-Risk-Prediction"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tuned_model.joblib")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "target_encoder.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "X_model_input.csv")

# --- 3. Feature Map ---
# (Keep the FEATURE_MAP dictionary exactly as it was before)
FEATURE_MAP = {
    # Personal Details
    'AGE': {'label': 'Age', 'tooltip': "Applicant's age."},
    'NETMONTHLYINCOME': {'label': 'Net Monthly Income', 'tooltip': "Applicant's take-home pay per month."},
    'Time_With_Curr_Empr': {'label': 'Time with Current Employer (Months)', 'tooltip': "Total months working at their current job."},
    'EDUCATION': {'label': 'Education Level', 'tooltip': "Applicant's highest completed education."},
    'MARITALSTATUS': {'label': 'Marital Status', 'tooltip': "Applicant's marital status."},
    'GENDER': {'label': 'Gender', 'tooltip': "Applicant's gender."},

    # Enquiry History
    'enq_L6m': {'label': 'Enquiries (Last 6 Months)', 'tooltip': 'Total number of credit enquiries in the last 6 months.'},
    'enq_L3m': {'label': 'Enquiries (Last 3 Months)', 'tooltip': 'Total number of credit enquiries in the last 3 months.'},
    'time_since_recent_enq': {'label': 'Days Since Recent Enquiry', 'tooltip': 'Number of days that have passed since the last credit enquiry.'},
    'CC_enq': {'label': 'Total Credit Card Enquiries', 'tooltip': 'Total number of enquiries for a credit card (ever).'},
    'CC_enq_L6m': {'label': 'Credit Card Enquiries (Last 6 Months)', 'tooltip': 'Number of credit card enquiries in the last 6 months.'},
    'PL_enq': {'label': 'Total Personal Loan Enquiries', 'tooltip': 'Total number of enquiries for a personal loan (ever).'},
    'PL_enq_L6m': {'label': 'Personal Loan Enquiries (Last 6 Months)', 'tooltip': 'Number of personal loan enquiries in the last 6 months.'},
    'last_prod_enq2': {'label': 'Last Product Enquired For', 'tooltip': 'The type of product the applicant last enquired about.'},
    'first_prod_enq2': {'label': 'First Product Enquired For', 'tooltip': 'The type of product the applicant first enquired about.'},

    # Loan Account History
    'Total_TL_opened_L12M': {'label': 'Accounts Opened (Last 12 Months)', 'tooltip': 'Total number of new loan accounts opened in the last 12 months.'},
    'Total_TL_opened_L6M': {'label': 'Accounts Opened (Last 6 Months)', 'tooltip': 'Total number of new loan accounts opened in the last 6 months.'},
    'Tot_TL_closed_L12M': {'label': 'Accounts Closed (Last 12 Months)', 'tooltip': 'Total number of loan accounts closed in the last 12 months.'},
    'Tot_TL_closed_L6M': {'label': 'Accounts Closed (Last 6 Months)', 'tooltip': 'Total number of loan accounts closed in the last 6 months.'},
    'pct_tl_open_L12M': {'label': 'Percent Accounts Opened (Last 12 Months)', 'tooltip': 'Percentage of total accounts that were opened in the last 12 months.'},
    'pct_tl_open_L6M': {'label': 'Percent Accounts Opened (Last 6 Months)', 'tooltip': 'Percentage of total accounts that were opened in the last 6 months.'},
    'pct_tl_closed_L12M': {'label': 'Percent Accounts Closed (Last 12 Months)', 'tooltip': 'Percentage of total accounts that were closed in the last 12 months.'},
    'pct_tl_closed_L6M': {'label': 'Percent Accounts Closed (Last 6 Months)', 'tooltip': 'Percentage of total accounts that were closed in the last 6 months.'},
    'Tot_Missed_Pmnt': {'label': 'Total Missed Payments (Ever)', 'tooltip': 'Total number of payments missed across all accounts.'},
    'Age_Oldest_TL': {'label': 'Age of Oldest Account (Months)', 'tooltip': 'How many months ago the applicant opened their very first loan account.'},
    'Age_Newest_TL': {'label': 'Age of Newest Account (Months)', 'tooltip': 'How many months ago the applicant opened their most recent loan account.'},
    'pct_opened_TLs_L6m_of_L12m': {'label': '% of 12-Month Opens in Last 6 Months', 'tooltip': 'Of the accounts opened in the last year, what percent were opened in the last 6 months.'},

    # Loan Portfolio Mix
    'CC_Flag': {'label': 'Has Credit Card Flag', 'tooltip': 'Does the applicant have a credit card? (1=Yes, 0=No)'},
    'PL_Flag': {'label': 'Has Personal Loan Flag', 'tooltip': 'Does the applicant have a personal loan? (1=Yes, 0=No)'},
    'HL_Flag': {'label': 'Has Housing Loan Flag', 'tooltip': 'Does the applicant have a housing loan? (1=Yes, 0=No)'},
    'GL_Flag': {'label': 'Has Gold Loan Flag', 'tooltip': 'Does the applicant have a gold loan? (1=Yes, 0=No)'},
    'Auto_TL': {'label': 'Count of Auto Loans', 'tooltip': 'Total number of automobile loans.'},
    'CC_TL': {'label': 'Count of Credit Card Accounts', 'tooltip': 'Total number of credit card accounts.'},
    'Consumer_TL': {'label': 'Count of Consumer Loans', 'tooltip': 'Total number of consumer goods loans.'},
    'Gold_TL': {'label': 'Count of Gold Loans', 'tooltip': 'Total number of gold loans.'},
    'Home_TL': {'label': 'Count of Home Loans', 'tooltip': 'Total number of housing loans.'},
    'PL_TL': {'label': 'Count of Personal Loans', 'tooltip': 'Total number of personal loans.'},
    'Other_TL': {'label': 'Count of Other Loans', 'tooltip': 'Total number of other types of loans.'},
}


# --- 4. Load Assets (Cached) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    except FileNotFoundError:
        st.error("Error: Asset files not found. Please check paths.")
        return None, None

@st.cache_data
def load_reference_data():
    """Loads reference data for column order, defaults, and options."""
    try:
        df = pd.read_csv(DATA_PATH, index_col='PROSPECTID')
        final_38_features = [col for col in FEATURE_MAP.keys() if col in df.columns]
        education_map = ['OTHERS', 'SSC', '12TH', 'UNDER GRADUATE', 'GRADUATE', 'POST-GRADUATE', 'PROFESSIONAL']
        nominal_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        defaults = {}
        options = {'EDUCATION': education_map}
        for col in final_38_features:
            if col in nominal_cols:
                defaults[col] = df[col].mode()[0]
                options[col] = df[col].unique().tolist()
            elif col == 'EDUCATION':
                defaults[col] = df[col].mode()[0]
            elif np.issubdtype(df[col].dtype, np.integer):
                defaults[col] = int(df[col].median())
            elif np.issubdtype(df[col].dtype, np.floating):
                defaults[col] = float(df[col].median())
        ordered_features = (
            list(FEATURE_MAP.keys())[0:6] + list(FEATURE_MAP.keys())[6:15] +
            list(FEATURE_MAP.keys())[15:27] + list(FEATURE_MAP.keys())[27:38]
        )
        return ordered_features, defaults, options
    except FileNotFoundError:
        st.error(f"Error: Reference data file not found at {DATA_PATH}")
        return None, None, None

model, encoder = load_assets()
ref_columns, ref_defaults, ref_options = load_reference_data()

# --- 5. Page Content ---
st.title("👤 Single Applicant Prediction")
st.write("Enter the details for a single applicant to get an immediate risk prediction.")

if ref_columns is None or model is None or encoder is None:
    st.error("Application cannot start. Critical assets failed to load.")
else:
    # Create a form for user input
    with st.form("manual_form"):
        st.info("Please fill out all features for an accurate prediction. Defaults are based on the median/mode of the training data.")
        input_data = {}

        # --- Create Logical Groups ---
        with st.expander("**1. Applicant's Personal Details**", expanded=True):
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(ref_columns[0:6]):
                with [col1, col2, col3][i % 3]:
                    if col in ref_options:
                        input_data[col] = st.selectbox(
                            label=FEATURE_MAP[col]['label'], options=ref_options[col],
                            index=ref_options[col].index(ref_defaults[col]), help=FEATURE_MAP[col]['tooltip']
                        )
                    else: # Numerical
                        is_float = isinstance(ref_defaults[col], float)
                        input_data[col] = st.number_input(
                            label=FEATURE_MAP[col]['label'], value=ref_defaults[col],
                            help=FEATURE_MAP[col]['tooltip'],
                            # --- FIX: Set step=None for floats, step=1 for ints ---
                            step=None if is_float else 1,
                            format="%.2f" if is_float else None
                        )

        with st.expander("**2. Recent Enquiry History**"):
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(ref_columns[6:15]):
                 with [col1, col2, col3][i % 3]:
                    if col in ref_options:
                        input_data[col] = st.selectbox(
                            label=FEATURE_MAP[col]['label'], options=ref_options[col],
                            index=ref_options[col].index(ref_defaults[col]), help=FEATURE_MAP[col]['tooltip']
                        )
                    else:
                        is_float = isinstance(ref_defaults[col], float)
                        input_data[col] = st.number_input(
                            label=FEATURE_MAP[col]['label'], value=ref_defaults[col],
                            help=FEATURE_MAP[col]['tooltip'],
                            step=None if is_float else 1,
                            format="%.2f" if is_float else None
                        )

        with st.expander("**3. Loan Account History**"):
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(ref_columns[15:27]):
                with [col1, col2, col3][i % 3]:
                    is_float = isinstance(ref_defaults[col], float)
                    input_data[col] = st.number_input(
                        label=FEATURE_MAP[col]['label'], value=ref_defaults[col],
                        help=FEATURE_MAP[col]['tooltip'],
                        step=None if is_float else 1,
                        format="%.2f" if is_float else None
                    )

        with st.expander("**4. Current Loan Portfolio Mix**"):
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(ref_columns[27:38]):
                 with [col1, col2, col3][i % 3]:
                    is_float = isinstance(ref_defaults[col], float)
                    input_data[col] = st.number_input(
                        label=FEATURE_MAP[col]['label'], value=ref_defaults[col],
                        help=FEATURE_MAP[col]['tooltip'],
                        step=None if is_float else 1,
                        format="%.2f" if is_float else None
                    )

        st.divider()
        # --- Submit Button (MUST be inside the form) ---
        submitted = st.form_submit_button("Predict Risk Category", type="primary")

    # --- Process the form submission (outside the form) ---
    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            # Re-order DF to match the model's training order (using the original keys)
            input_df = input_df[list(FEATURE_MAP.keys())]

            # --- Run Prediction ---
            prediction_encoded = model.predict(input_df)
            prediction_decoded = encoder.inverse_transform(prediction_encoded)[0]

            # --- Display Result ---
            st.subheader("Prediction Result")
            # (Display logic remains the same)
            if prediction_decoded == 'P1':
                 st.success(f"## ✅ Predicted Risk Category: P1 (Lowest Risk)")
                 st.markdown("This applicant has a very strong profile and is a prime candidate for approval.")
            elif prediction_decoded == 'P2':
                 st.success(f"## ✔️ Predicted Risk Category: P2 (Low Risk)")
                 st.markdown("This applicant has a good profile and is a strong candidate for approval.")
            elif prediction_decoded == 'P3':
                 st.warning(f"## ⚠️ Predicted Risk Category: P3 (Moderate Risk)")
                 st.markdown("This applicant shows some risk factors. Recommend manual review or approval with caution.")
            elif prediction_decoded == 'P4':
                 st.error(f"## ❌ Predicted Risk Category: P4 (Highest Risk)")
                 st.markdown("This applicant has a very high-risk profile. Recommend rejection.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)