# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib


st.title("Hello, Streamlit!")
st.write("✅ App is working if you see this!")


# Load trained model
model = joblib.load("model.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Fill the following inputs to predict if the customer will churn.")

# Example inputs (customize based on your dataset)
tenure = st.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 50)
total_charges = st.slider("Total Charges ($)", 0, 10000, 1000)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Manual encoding (you must match training data structure!)
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# Make prediction
input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract_One year": [contract_one_year],
    "Contract_Two year": [contract_two_year]
})

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
