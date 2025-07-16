# streamlit_app.py (Based on final trained columns)

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")

# Load trained model pipeline
model = joblib.load("model.pkl")

# Collect user inputs
st.header("🔧 Customer Information")

user_input = {
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "ContractType": st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "CustomerID": st.text_input("Customer ID", value="12345-A"),
    "Age": st.number_input("Age", min_value=18, max_value=100, value=35),
    "Tenure": st.slider("Tenure (months)", 0, 72, 12),
    "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0),
    "TotalCharges": st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)
}

input_df = pd.DataFrame([user_input])

# Make prediction
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("🔎 Prediction Result:")
        if prediction == 1:
            st.error("❌ This customer is likely to churn.")
        else:
            st.success("✅ This customer is not likely to churn.")
    except Exception as e:
        st.error("⚠️ Prediction failed:")
        st.code(str(e))
        st.write("Check your input data matches model expectations.")
