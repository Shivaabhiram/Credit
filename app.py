import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Loan Default Prediction System")
st.write("Predict whether a loan applicant is risky or safe")

input = []
for feature in features:
    value = st.number_input(f"Enter {feature}", value=0.0)
    input.append(value)

if st.button("Predict Loan risk"):
    input_array = np.array(input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.success("Low Risk")
    else:
        st.error("High Risk")

    Education_map = {
        "Bachelor's": 0,
        "Master's":1,
        "High School":2,
        "PhD":3
    }

    EmploymentType_map = {
        "Full-time":0,
        "Unemployed":1
    }

    MartialStatus_map = {
        "Divorced":0,
        "Married":1,
        "Single":2
    }

    HasMortgage_map = {
        "Yes":1,
        "No":0
    }

    HasDependents_map = {
        "Yes":1,
        "No":0
    }

    LoanPurpose_map = {
        "Other":0,
        "Auto":1,
        "Business":2,
        "Home":3,
        "Education":4
    }

    HasCoSigner_map = {
        "Yes":1,
        "No":0
    }

    Education_input = st.selectbox(
        "Education",
        list(Education_map.keys())
    )

    EmploymentType_input = st.selectbox(
        "EmploymentType",
        list(EmploymentType_map.keys())
    )

    MartialStatus_input = st.selectbox(
        "MartialStatus",
        list(MartialStatus_map.keys())
    )

    HasMortgage_input = st.selectbox(
        "HasMortgage",
        list(HasMortgage_map.keys())
    )

    HasDependents_input = st.selectbox(
        "HasDependents",
        list(HasDependents_map.keys())
    )

    LoanPurpose_input = st.selectbox(
        "LoanPurpose",
        list(LoanPurpose_map.keys())
    )

    HasCosigner_input = st.selectbox(
        "HasCosigner",
        list(HasCoSigner_map.keys())
    )

    Education_encoded = Education_map[Education_input]
    EmploymentType_encoded = EmploymentType_map[EmploymentType_input]
    MartialStatus_encoded = MartialStatus_map[MartialStatus_input]
    HasMortgage_encoded = HasMortgage_map[HasMortgage_input]
    HasDependents_encoded = HasDependents_map[HasDependents_input]
    LoanPurpose_encoded = LoanPurpose_map[LoanPurpose_input]
    HasCosigner_encoded = HasCoSigner_map[HasCosigner_input]


    input_data = pd.DataFrame([{
        "Education":Education_encoded,
        "EmploymentType":EmploymentType_encoded,
        "MartialStatus":MartialStatus_encoded,
        "HasMortgage":HasMortgage_encoded,
        "HasDependents":HasDependents_encoded,
        "LoanPurpose":LoanPurpose_encoded,
        "HasCoSigner":HasCosigner_encoded
    }])

    prediction = model.predict(input_data)