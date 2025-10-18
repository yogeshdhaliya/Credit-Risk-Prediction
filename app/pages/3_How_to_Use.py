import streamlit as st

st.set_page_config(
    page_title="Credit Risk | How to Use",
    page_icon="❓",
    layout="wide"
)

st.title("❓ How to Use This App")
st.info(
    "This application is a tool for assessing loan applicant risk. "
    "Use the navigation sidebar to select a page."
)

st.divider()

st.header("Home")
st.markdown("""
The **Home** page serves as the main dashboard, providing two key insights:

1.  **Target Class Distribution:** This pie chart shows the original data's severe class imbalance. Our model was built specifically to handle this, using **SMOTE** (Synthetic Minority Over-sampling Technique) to learn from the rare P1, P3, and P4 classes.
2.  **Model Feature Importance:** This bar chart shows the Top 15 features the model uses to make its decisions. This is crucial for **explainability**, proving our model uses logical data points (like enquiry history, income, and account age) to determine risk.
""")

st.header("1. Batch Prediction")
st.markdown("""
This page is for processing many applicants at once.

1.  **Prepare your file:** Ensure your `.csv` file contains all **38 required feature columns**. The column names must *exactly* match the ones used in training (e.g., `NETMONTHLYINCOME`, `enq_L6m`, `Age_Oldest_TL`). You can use the "Single Applicant Prediction" page as a reference for all 38 required features.
2.  **Upload:** Click "Browse files" and select your CSV.
3.  **Process:** The app will run predictions for every row in your file.
4.  **Review Results:** After processing, you will see a pie chart of the predicted risk distribution for your batch and a detailed table with the prediction for each applicant.
5.  **Download:** You can download this results table as a new CSV.
""")

st.header("2. Single Applicant Prediction")
st.markdown("""
This page is for getting an immediate prediction for one person.

1.  **Fill the Form:** The form is grouped into four sections (Personal, Enquiry, Account History, Portfolio). Fill out all 38 features.
    * **Need help?** Hover over the **(?)** icon next to any input to get a detailed description of that feature.
    * The form is pre-filled with *default values* (the median or mode from the training data) to make it easier to test.
2.  **Predict:** Click the "Predict Risk Category" button at the bottom.
3.  **Review Result:** The app will instantly show you the predicted risk category and a recommended action.
""")

st.header("Interpreting the Results (The 4 Categories)")
st.markdown("""
The model assigns one of four risk categories. Here is what they mean:

* **P1 (Lowest Risk):** A premium applicant. They have the strongest profiles and highest likelihood to repay.
* **P2 (Low Risk):** A good, standard applicant. These are the "bread and butter" customers who are reliable and meet all standard criteria.
* **P3 (Moderate Risk):** A subprime applicant. They show some warning signs (e.g., high enquiries, some missed payments) but are not a definite loss. Our tuning in Phase 5 was focused on getting better at identifying this specific group.
* **P4 (Highest Risk):** A high-risk applicant. These applicants show multiple strong indicators of default risk.

**As a general guide (your request):** The model has learned that applicants with a combination of **low income**, **high recent credit enquiries**, and a **history of missed payments** are much more likely to be classified as **P4 (Highest Risk)**.
""")