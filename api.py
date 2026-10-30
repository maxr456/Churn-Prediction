"""
api.py
Production REST API for Customer Churn Prediction using FastAPI.
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API predicting churn risk using Random Forest and tuned decision thresholds.",
    version="1.0.0"
)

# 2. Load artifacts on startup
MODEL_PATH = os.path.join("models", "rf_model.joblib")
FEATURES_PATH = os.path.join("models", "model_features.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Model artifacts not found! Run train_model.py first.")

model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)

# Tuned threshold from Stage 6 evaluation (>85% recall)
DECISION_THRESHOLD = 0.20

# 3. Define the input data schema with realistic defaults
class CustomerData(BaseModel):
    gender: str = Field(default="Male", description="Gender (Male / Female)")
    age: int = Field(default=45, ge=18, le=100)
    country: str = Field(default="Canada")
    city: str = Field(default="Toronto")
    customer_segment: str = Field(default="Individual", description="Individual, SME, or Enterprise")
    tenure_months: int = Field(default=12, ge=0, description="Customer relationship length in months")
    signup_channel: str = Field(default="Web", description="Web, Mobile, or Referral")
    contract_type: str = Field(default="Monthly", description="Monthly, Quarterly, or Yearly")
    monthly_logins: int = Field(default=15, ge=0)
    weekly_active_days: int = Field(default=3, ge=0, le=7)
    avg_session_time: float = Field(default=15.0, ge=0.0)
    features_used: int = Field(default=5, ge=0)
    usage_growth_rate: float = Field(default=0.0)
    last_login_days_ago: int = Field(default=5, ge=0)
    monthly_fee: int = Field(default=35, ge=0)
    total_revenue: int = Field(default=420, ge=0)
    payment_method: str = Field(default="Card", description="Card, Bank Transfer, or PayPal")
    payment_failures: int = Field(default=0, ge=0)
    discount_applied: str = Field(default="No", description="Yes or No")
    price_increase_last_3m: str = Field(default="No", description="Yes or No")
    support_tickets: int = Field(default=1, ge=0)
    avg_resolution_time: float = Field(default=24.0, ge=0.0)
    complaint_type: str = Field(default="No Complaint", description="Billing, Service, Technical, or No Complaint")
    csat_score: float = Field(default=4.0, ge=1.0, le=5.0, description="Customer satisfaction score (1.0 to 5.0)")
    escalations: int = Field(default=0, ge=0)
    email_open_rate: float = Field(default=0.50, ge=0.0, le=1.0)
    marketing_click_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    nps_score: int = Field(default=20, ge=-100, le=100)
    survey_response: str = Field(default="Satisfied", description="Satisfied, Neutral, or Unsatisfied")
    referral_count: int = Field(default=1, ge=0)

# 4. Root Health Check Endpoint
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "docs_url": "/docs",
        "decision_threshold": DECISION_THRESHOLD
    }

# 5. Prediction Endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert input to DataFrame
        input_dict = customer.model_dump()
        df_input = pd.DataFrame([input_dict])

        # One-Hot Encode and reindex to align with training features
        df_encoded = pd.get_dummies(df_input, drop_first=True, dtype=int)
        df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

        # Predict churn probability
        churn_prob = float(model.predict_proba(df_aligned)[0, 1])

        # Apply calibrated business threshold
        is_churn_risk = churn_prob >= DECISION_THRESHOLD

        if churn_prob >= 0.35:
            risk_level = "High"
            action = "Immediate retention action required: CSAT review & discount offer."
        elif churn_prob >= DECISION_THRESHOLD:
            risk_level = "Medium"
            action = "Proactive check-in: monitor usage and resolve open tickets."
        else:
            risk_level = "Low"
            action = "Healthy customer: standard engagement."

        return {
            "churn_prediction": int(is_churn_risk),
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "threshold_used": DECISION_THRESHOLD,
            "recommended_action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

