# streamlit_app_new.py

import streamlit as st
import pandas as pd
import joblib

st.title("📊 Customer Churn Prediction App")
st.write("Fill the following inputs to predict if the customer will churn.")

# Load trained model pipeline
model = joblib.load("model.pkl")

# Collect user inputs
st.header("🔧 Customer Information")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
total_charges = st.slider("Total Charges ($)", 0, 10000, 1000)
age = st.slider("Age", 0, 100)
InternetService = st.selectbox("Internet Service", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
gender = st.selectbox("Gender", ["Male", "Female"])



# Create input DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "SeniorCitizen": [0],  # optional default
    "tenure": [tenure],
    "Contract": [contract],
    "InternetService": [InternetService],
    "TechSupport": [TechSupport],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
