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