
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦"
)

@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

saved = load_model()
model = saved["model"]
scaler = saved["scaler"]
feature_columns = saved["feature_columns"]

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details below.")

with st.form("loan_form"):
    st.subheader("Applicant Information")

    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=50000.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=20, value=0)
    existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
    savings = st.number_input("Savings", min_value=0.0, value=20000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=100000.0)
    loan_term = st.number_input("Loan Term", min_value=1, value=12)

    education = st.selectbox(
        "Education Level", ["Graduate", "Not Graduate"]
    )
    employment = st.selectbox(
        "Employment Status", ["Salaried", "Self-employed", "Unemployed"]
    )
    marital = st.selectbox("Marital Status", ["Single", "Married"])
    purpose = st.selectbox(
        "Loan Purpose", ["Car", "Education", "Home", "Personal"]
    )
    property_area = st.selectbox(
        "Property Area", ["Urban", "Semiurban", "Rural"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    employer = st.selectbox(
        "Employer Category",
        ["Government", "MNC", "Private", "Unemployed"]
    )

    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, value=0.3)
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=700
    )

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    row = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education,
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer,
        "DTI_Ratio_sq": dti_ratio ** 2,
        "Credit_Score_sq": credit_score ** 2,
    }

    input_df = pd.DataFrame([row])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.success("Prediction: Loan Approved ✅")
    else:
        st.error("Prediction: Loan Not Approved ❌")
