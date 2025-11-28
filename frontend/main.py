import streamlit as st
import requests
import pandas as pd
import os

# Sayfa Ayarları
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="🔮",
    layout="wide"
)

# API Address
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Title and Description
st.title("🔮 Telco Customer Churn Prediction System")
st.markdown("""
This system calculates customer churn probability using advanced MLOps architecture (XGBoost + MLflow + FastAPI).
""")

# --- LEFT MENU (INPUTS) ---
st.sidebar.header("Customer Information")


def user_input_features():
    # Categorical Variables
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen?", (0, 1),
                                          format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Does He Have a Wife?/Husband", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependent Person?", ("Yes", "No"))

    st.sidebar.markdown("---")

    # Services
    tenure = st.sidebar.slider("How Many Months Have You Been a Customer? (Tenure)", 1, 72, 12)
    phone_service = st.sidebar.selectbox("Is There a Phone Line?", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines?", ("Yes", "No", "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))

    online_security = st.sidebar.selectbox("Online Security?", ("Yes", "No", "No internet service"))
    online_backup = st.sidebar.selectbox("Backup Service?", ("Yes", "No", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection?", ("Yes", "No", "No internet service"))
    tech_support = st.sidebar.selectbox("Technical Support?", ("Yes", "No", "No internet service"))

    streaming_tv = st.sidebar.selectbox("Broadcast TV?", ("Yes", "No", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Movie Streaming?", ("Yes", "No", "No internet service"))

    st.sidebar.markdown("---")

    # Payment Information
    contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Invoice?", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Payment method", (
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))

    monthly_charges = st.sidebar.number_input("Monthly Bill ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Invoice ($)", min_value=0.0, value=monthly_charges * tenure)

    # Generate JSON data
    data = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure,
        "phoneservice": phone_service,
        "multiplelines": multiple_lines,
        "internetservice": internet_service,
        "onlinesecurity": online_security,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        "techsupport": tech_support,
        "streamingtv": streaming_tv,
        "streamingmovies": streaming_movies,
        "contract": contract,
        "paperlessbilling": paperless_billing,
        "paymentmethod": payment_method,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges
    }
    return data


input_data = user_input_features()

# --- MIDDLE FIELD (RESULTS DISPLAY) ---

# Show Entries
st.subheader("Selected Customer Profile")
st.json(input_data, expanded=False)

# Buton
if st.button("🚀 PERFORM RISK ANALYSIS", type="primary"):

    with st.spinner("Model (XGBoost) is analyzing..."):
        try:
            # Make a request to the API
            response = requests.post(API_URL, json=input_data)

            if response.status_code == 200:
                result = response.json()
                churn_prob = result["churn_probability"]
                churn_bool = result["prediction"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Possibility of Abandonment", value=f"%{churn_prob * 100:.2f}")

                with col2:
                    if churn_bool == 1:
                        st.error("⚠️ RISKY CUSTOMER! (Churn)")
                        st.write("This customer should be offered an urgent discount.")
                    else:
                        st.success("✅ SECURE CUSTOMER (Loyal)")
                        st.write("Customer satisfaction appears to be high.")

                # Probability Bar
                st.progress(churn_prob)

            else:
                st.error(f"Error occurred: {response.text}")

        except Exception as e:
            st.error(f"API Connection Error: {e}")
            st.info("Make sure the API (uvicorn) is running.")