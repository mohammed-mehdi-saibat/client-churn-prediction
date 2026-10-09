import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Custom Header Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Telecom Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive dashboard powered by your optimized XGBoost model.</div>', unsafe_allow_html=True)

# Load the trained model safely
@st.cache_resource
def load_model():
    model_paths = [
        "../models/classification/churn_model.pkl",
        "models/classification/churn_model.pkl",
        "churn_model.pkl"
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file (`churn_model.pkl`) not found! Ensure it is located in `models/classification/`.")
else:
    # Sidebar inputs for customer profile
    st.sidebar.header("Customer Profile Inputs")

    def user_input_features():
        senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=70.0)
        total_charges = tenure * monthly_charges
        
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
        dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
        
        phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
        internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_t_v = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.sidebar.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        
        # Engineered features matching training schema
        total_services_count = sum([
            1 if x == "Yes" else 0 for x in [
                phone_service, multiple_lines, online_security, online_backup, 
                device_protection, tech_support, streaming_t_v, streaming_movies
            ]
        ])
        
        avg_monthly_spend_ratio = monthly_charges / (total_charges / max(tenure, 1)) if tenure > 0 else 1.0
        
        cluster_1, cluster_2, cluster_3 = 1, 0, 0

        data = {
            "senior_citizen": senior_citizen,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "total_services_count": total_services_count,
            "avg_monthly_spend_ratio": avg_monthly_spend_ratio,
            "cluster_1": cluster_1,
            "cluster_2": cluster_2,
            "cluster_3": cluster_3,
            "gender": gender,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_t_v": streaming_t_v,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "tenure_group": "1-12_months" if tenure <= 12 else ("13-24_months" if tenure <= 24 else "25-48_months")
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    # Layout Setup
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Profile Overview")
        st.dataframe(input_df.T, use_container_width=True, height=400)

    with col2:
        st.subheader("Prediction Dashboard")
        if st.button("Run Churn Prediction", type="primary", use_container_width=True):
            try:
                # Convert categorical columns to pandas 'category' dtype to satisfy XGBoost requirements
                categorical_cols = [
                    "gender", "partner", "dependents", "phone_service", "multiple_lines", 
                    "internet_service", "online_security", "online_backup", "device_protection", 
                    "tech_support", "streaming_t_v", "streaming_movies", "contract", 
                    "paperless_billing", "payment_method", "tenure_group"
                ]
                for col in categorical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype("category")

                # Ensure feature order matches model booster feature names if available
                if hasattr(model, "feature_names_in_"):
                    # Reindex columns to match training feature order exactly
                    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
                    for c in missing_cols:
                        input_df[c] = 0  # fallback default if any extra feature was expected
                    input_df = input_df[model.feature_names_in_]

                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[:, 1][0]
                
                if prediction[0] == 1:
                    st.error(f"⚠️ **High Churn Risk Detected!**")
                    st.markdown("💡 *Recommendation:* Consider offering a loyalty incentive or discount contract.")
                else:
                    st.success(f"✅ **Customer Appears Stable (Low Risk)**")
                    st.markdown("💡 *Recommendation:* Maintain standard engagement practices.")
                    
                st.metric(label="Predicted Churn Probability", value=f"{probability:.2%}")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")