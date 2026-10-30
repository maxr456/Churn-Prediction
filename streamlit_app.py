"""
streamlit_app.py
Interactive Web Dashboard for Customer Churn Prediction using Streamlit.
"""
import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="👥",
    layout="wide"
)

# 2. Load ML Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join("models", "rf_model.joblib"))
    features = joblib.load(os.path.join("models", "model_features.joblib"))
    return model, features

try:
    model, expected_features = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run 'python train_model.py' first.")
    st.stop()

DECISION_THRESHOLD = 0.20

# 3. Header & Project Intro
st.title("👥 Customer Churn Prediction & Retention Dashboard")
st.markdown("""
Predict customer churn probability in real time using our tuned **Random Forest** model.
Adjust customer characteristics below to simulate risk profiles and view recommended retention strategies.
""")

st.divider()

# 4. Input Form across 3 visual columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Usage & Engagement")
    tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12,
                              help="Top Churn Driver: Shorter tenure correlates with higher churn.")
    monthly_logins = st.slider("Monthly Logins", min_value=0, max_value=40, value=18)
    weekly_active_days = st.slider("Weekly Active Days", min_value=0, max_value=7, value=3)
    last_login_days_ago = st.slider("Last Login (Days Ago)", min_value=0, max_value=30, value=4)
    features_used = st.number_input("Features Used", min_value=1, max_value=10, value=5)
    customer_segment = st.selectbox("Customer Segment", ["Individual", "SME", "Enterprise"])
    contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"])
    signup_channel = st.selectbox("Signup Channel", ["Web", "Mobile", "Referral"])

with col2:
    st.subheader("2. Satisfaction & Support")
    csat_score = st.slider("CSAT Score (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=4.0, step=0.1,
                           help="Top Churn Driver: Scores below 3.0 trigger immediate churn risk.")
    complaint_type = st.selectbox("Complaint Category", ["No Complaint", "Billing", "Service", "Technical"])
    survey_response = st.selectbox("Survey Response", ["Satisfied", "Neutral", "Unsatisfied"])
    nps_score = st.slider("NPS Score (-100 to 100)", min_value=-100, max_value=100, value=20)
    support_tickets = st.number_input("Support Tickets Logged", min_value=0, max_value=20, value=1)
    avg_resolution_time = st.number_input("Avg Resolution Time (Hours)", min_value=1.0, max_value=100.0, value=24.0)
    escalations = st.number_input("Escalations", min_value=0, max_value=5, value=0)

with col3:
    st.subheader("3. Financials & Account")
    monthly_fee = st.number_input("Monthly Fee ($)", min_value=10, max_value=200, value=35)
    total_revenue = st.number_input("Total Revenue ($)", min_value=0, max_value=10000, value=420)
    payment_failures = st.slider("Payment Failures", min_value=0, max_value=10, value=0,
                                 help="Top Churn Driver: Payment failures lead to involuntary churn.")
    payment_method = st.selectbox("Payment Method", ["Card", "Bank Transfer", "PayPal"])
    price_increase_last_3m = st.radio("Price Increase in Last 3 Months?", ["No", "Yes"], horizontal=True)
    discount_applied = st.radio("Discount Applied Currently?", ["No", "Yes"], horizontal=True)
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=42)
    gender = st.selectbox("Gender", ["Male", "Female"])

# Default non-critical variables
country = "Canada"
city = "Toronto"
usage_growth_rate = 0.0
email_open_rate = 0.50
marketing_click_rate = 0.25
referral_count = 1

st.divider()

# 5. Prediction Execution
if st.button("🔮 Predict Customer Churn Risk", type="primary", use_container_width=True):
    # Assemble input record
    customer_dict = {
        'gender': gender, 'age': age, 'country': country, 'city': city,
        'customer_segment': customer_segment, 'tenure_months': tenure_months,
        'signup_channel': signup_channel, 'contract_type': contract_type,
        'monthly_logins': monthly_logins, 'weekly_active_days': weekly_active_days,
        'avg_session_time': 15.0, 'features_used': features_used,
        'usage_growth_rate': usage_growth_rate, 'last_login_days_ago': last_login_days_ago,
        'monthly_fee': monthly_fee, 'total_revenue': total_revenue,
        'payment_method': payment_method, 'payment_failures': payment_failures,
        'discount_applied': discount_applied, 'price_increase_last_3m': price_increase_last_3m,
        'support_tickets': support_tickets, 'avg_resolution_time': avg_resolution_time,
        'complaint_type': complaint_type, 'csat_score': csat_score,
        'escalations': escalations, 'email_open_rate': email_open_rate,
        'marketing_click_rate': marketing_click_rate, 'nps_score': nps_score,
        'survey_response': survey_response, 'referral_count': referral_count
    }

    # Transform input
    df_input = pd.DataFrame([customer_dict])
    df_encoded = pd.get_dummies(df_input, drop_first=True, dtype=int)
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

    # Compute probability
    churn_prob = float(model.predict_proba(df_aligned)[0, 1])

    # Display Results
    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Predicted Churn Probability",
            value=f"{churn_prob * 100:.1f}%",
            delta=f"{(churn_prob - DECISION_THRESHOLD) * 100:+.1f}% vs Threshold (20%)",
            delta_color="inverse"
        )

    with res_col2:
        if churn_prob >= DECISION_THRESHOLD:
            st.error(f"⚠️ **HIGH CHURN RISK DETECTED** (Probability $\\ge$ {DECISION_THRESHOLD*100:.0f}%)")
            st.markdown("""
            **Recommended Immediate Actions:**
            * 📞 Schedule a proactive outreach from Customer Success.
            * 💳 Verify payment methods if payment failures occurred.
            * 🎁 Offer a targeted retention discount or feature review.
            """)
        else:
            st.success(f"✅ **CUSTOMER IS HEALTHY / LOW CHURN RISK** (Probability < {DECISION_THRESHOLD*100:.0f}%)")
            st.markdown("""
            **Recommended Action:**
            * Maintain standard engagement and monitor periodic CSAT surveys.
            """)

