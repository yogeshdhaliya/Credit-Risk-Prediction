# --- IMPORTS MUST BE FIRST ---
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import io
import numpy as np # <-- FIX: Import numpy
import pathlib
# --- 1. Configuration ---
st.set_page_config(
    page_title="Credit Risk | Batch Prediction",
    page_icon="📂",
    layout="wide"
)


# --- 2. Define File Paths (RELATIVE PATHS) ---
try:
    APP_DIR = pathlib.Path(__file__).parent
except NameError:
    # Handle cases where __file__ is not defined
    APP_DIR = pathlib.Path.cwd()

# Get the project root directory (two levels up from 'pages/')
PROJECT_ROOT = APP_DIR.parent.parent

# Define paths relative to the project root
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "target_encoder.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_model_input.csv" # Used for column names

# --- 3. Load Assets (Cached) ---
# Use decorators AFTER import st
@st.cache_resource
def load_assets():
    """Loads the saved model and target encoder."""
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    except FileNotFoundError:
        st.error("Error: Asset files not found. Please check paths.")
        return None, None

@st.cache_data
def load_reference_columns():
    """Loads the 38 feature column names."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df.columns.drop('PROSPECTID').tolist()
    except FileNotFoundError:
        st.error(f"Error: Reference data file not found at {DATA_PATH}")
        return None

# Use decorator AFTER import st
@st.cache_data
def create_sample_csv(column_order):
    """Creates an in-memory CSV file with 50 diverse sample data rows."""
    num_samples = 50
    np.random.seed(42) # Now np is defined
    # (Keep the expanded sample_data dictionary from the previous step here)
    sample_data = {
        'AGE': np.random.randint(21, 65, num_samples),
        'NETMONTHLYINCOME': np.random.randint(10000, 60000, num_samples),
        'Time_With_Curr_Empr': np.random.randint(6, 300, num_samples),
        'pct_opened_TLs_L6m_of_L12m': np.random.rand(num_samples),
        'CC_Flag': np.random.randint(0, 2, num_samples),
        'PL_Flag': np.random.randint(0, 2, num_samples),
        'HL_Flag': np.random.randint(0, 2, num_samples),
        'GL_Flag': np.random.randint(0, 2, num_samples),
        'Total_TL_opened_L6M': np.random.randint(0, 5, num_samples),
        'Tot_TL_closed_L6M': np.random.randint(0, 3, num_samples),
        'pct_tl_open_L6M': np.random.rand(num_samples) * 0.5,
        'pct_tl_closed_L6M': np.random.rand(num_samples) * 0.2,
        'Total_TL_opened_L12M': np.random.randint(0, 10, num_samples),
        'Tot_TL_closed_L12M': np.random.randint(0, 5, num_samples),
        'pct_tl_open_L12M': np.random.rand(num_samples),
        'pct_tl_closed_L12M': np.random.rand(num_samples) * 0.5,
        'Tot_Missed_Pmnt': np.random.randint(0, 10, num_samples),
        'Auto_TL': np.random.randint(0, 3, num_samples),
        'CC_TL': np.random.randint(0, 5, num_samples),
        'Consumer_TL': np.random.randint(0, 7, num_samples),
        'Gold_TL': np.random.randint(0, 2, num_samples),
        'Home_TL': np.random.randint(0, 2, num_samples),
        'PL_TL': np.random.randint(0, 4, num_samples),
        'Other_TL': np.random.randint(0, 3, num_samples),
        'Age_Oldest_TL': np.random.uniform(6.0, 240.0, num_samples),
        'Age_Newest_TL': np.random.uniform(1.0, 60.0, num_samples),
        'MARITALSTATUS': np.random.choice(['Married', 'Single'], num_samples, p=[0.7, 0.3]),
        'EDUCATION': np.random.choice(['GRADUATE', '12TH', 'SSC', 'UNDER GRADUATE', 'POST-GRADUATE', 'OTHERS', 'PROFESSIONAL'], num_samples, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]),
        'GENDER': np.random.choice(['M', 'F'], num_samples, p=[0.8, 0.2]),
        'last_prod_enq2': np.random.choice(['others', 'ConsumerLoan', 'PL', 'CC', 'AL', 'HL'], num_samples, p=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05]),
        'first_prod_enq2': np.random.choice(['others', 'ConsumerLoan', 'PL', 'AL', 'CC', 'HL'], num_samples, p=[0.5, 0.2, 0.1, 0.07, 0.07, 0.06]),
        'CC_enq': np.random.randint(0, 5, num_samples),
        'CC_enq_L6m': np.random.randint(0, 3, num_samples),
        'PL_enq': np.random.randint(0, 7, num_samples),
        'PL_enq_L6m': np.random.randint(0, 4, num_samples),
        'time_since_recent_enq': np.random.uniform(0.0, 730.0, num_samples),
        'enq_L6m': np.random.randint(0, 10, num_samples),
        'enq_L3m': np.random.randint(0, 6, num_samples)
    }
    # Constraints
    sample_data['CC_enq_L6m'] = np.minimum(sample_data['CC_enq_L6m'], sample_data['CC_enq'])
    sample_data['PL_enq_L6m'] = np.minimum(sample_data['PL_enq_L6m'], sample_data['PL_enq'])
    sample_data['enq_L3m'] = np.minimum(sample_data['enq_L3m'], sample_data['enq_L6m'])
    sample_data['Age_Newest_TL'] = np.minimum(sample_data['Age_Newest_TL'], sample_data['Age_Oldest_TL'])

    sample_df = pd.DataFrame(sample_data)
    sample_df = sample_df[column_order] # Ensure column order
    output = io.StringIO()
    sample_df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')

# --- Load assets AFTER defining functions ---
model, encoder = load_assets()
ref_columns = load_reference_columns()

# --- 4. Page Content ---
st.title("📂 Batch Applicant Prediction")
st.write("Upload a CSV file containing applicant data to get risk predictions for all rows.")

if model is None or encoder is None or ref_columns is None:
    st.error("Application cannot start. Critical assets failed to load.")
else:
    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    # --- Download Sample CSV Button ---
    st.write("Need an example file? Download a sample CSV with 50 rows and the required 38 columns:")
    # Ensure ref_columns is loaded before calling this
    if ref_columns:
        sample_csv_data = create_sample_csv(ref_columns)
        st.download_button(
            label="Download Sample CSV (50 Rows)",
            data=sample_csv_data,
            file_name="sample_applicants_50.csv",
            mime="text/csv",
            key='download-sample-csv'
        )
    else:
        st.warning("Could not generate sample CSV because reference columns failed to load.")
    st.divider()

    # --- Prediction Logic (only runs if file is uploaded) ---
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # (Keep the rest of the prediction and results display logic exactly as it was)
            # --- Validation ---
            missing_cols = set(ref_columns) - set(df.columns)
            if missing_cols:
                st.error(f"Upload Error: The following required columns are missing from your file: {', '.join(missing_cols)}")
                st.stop()
            # --- ID Handling ---
            if 'PROSPECTID' in df.columns:
                prospect_ids = df['PROSPECTID']
            else:
                prospect_ids = pd.Series(range(1, len(df) + 1), name="Row_ID")
                df['Row_ID'] = prospect_ids
                df = df.set_index('Row_ID')
                prospect_ids.index = df.index
            # --- Column Ordering ---
            try:
                df_ordered = df[ref_columns]
            except KeyError as e:
                st.error(f"Error re-ordering columns. Missing: {e}")
                st.stop()
            # --- Prediction ---
            with st.spinner("Processing... Running predictions on all rows."):
                predictions_encoded = model.predict(df_ordered)
                predictions_decoded = encoder.inverse_transform(predictions_encoded)
            st.success("✅ Predictions complete!")
            # --- Results DF ---
            results_df = pd.DataFrame({
                'Applicant_ID': prospect_ids.values,
                'Predicted_Risk_Category': predictions_decoded
            })
            # --- Display Results ---
            st.header("Prediction Results")
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.subheader("Prediction Distribution")
                st.write("This pie chart shows the breakdown of the predictions for your uploaded batch.")
                pred_counts = results_df['Predicted_Risk_Category'].value_counts().reset_index()
                pred_counts.columns = ['Risk Category', 'Count']
                fig_pie_pred = px.pie(
                    pred_counts, names='Risk Category', values='Count',
                    title=f'Prediction Results for {len(results_df)} Applicants', hole=0.3
                )
                fig_pie_pred.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie_pred, use_container_width=True)
            with col2:
                st.subheader("Prediction Details (Data)")
                st.write("See the predicted risk category for each applicant.")
                st.dataframe(results_df, use_container_width=True, height=350)
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')
                csv_results_data = convert_df_to_csv(results_df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_results_data,
                    file_name="risk_predictions.csv",
                    mime="text/csv",
                    key='download-results-csv'
                )
        except pd.errors.EmptyDataError:
             st.error("Upload Error: The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An unexpected error occurred during processing:")
            st.exception(e)