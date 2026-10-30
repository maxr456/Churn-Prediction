# Customer Churn Prediction: End-to-End ML Pipeline & Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/)

> **Live Web Application (Deployed):** 
> **[https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/](https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/)**

---

An end-to-end Machine Learning solution designed to predict customer churn risk, uncover behavioral churn drivers, and deliver retention insights.

This repository covers the complete lifecycle: from **Exploratory Data Analysis (EDA)** and **stratified preprocessing** to **model benchmarking**, **business threshold tuning**, and dual production deployment (**Streamlit Web Dashboard** and **FastAPI REST API**).

---

##  Executive Summary

* **Live Demo**: Deployed to production at [Streamlit Community Cloud](https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/).
* **Business Problem**: Identify customers at risk of churning early enough to enable proactive retention interventions.
* **Dataset**: 10,000 customer records with 32 engagement, financial, and support attributes.
* **Core Challenge**: **Class Imbalance** (~10.2% churn rate vs. ~89.8% retention rate).
* **Champion Model**: **Balanced Random Forest Classifier** with a calibrated decision threshold ($0.20$), achieving **>85% Recall** and **0.805 ROC-AUC**.

---

##  Model Performance & Benchmarking

In an imbalanced churn dataset, **Accuracy is a deceptive metric**—a naive model predicting "Retained" for every customer scores ~89.8% accuracy while detecting **0%** of churners. We benchmarked models using **Precision, Recall, F1-Score, and ROC-AUC**:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Baseline** | 89.80% | 0.00% | 0.00% | 0.0000 | 0.5000 | Fails completely: catches 0 churners. |
| **Logistic Regression (Balanced)** | 66.45% | 18.23% | 65.69% | 0.2854 | 0.7245 | High recall, but linear boundaries yield lower separation. |
| **Random Forest (Default Thresh 0.50)** | **89.00%** | **36.21%** | 10.29% | 0.1603 | **0.8054** | High overall ranking power (ROC-AUC), but default 0.5 cutoff misses risk. |
| **Random Forest (Tuned Thresh 0.20)** | 70.15% | 23.36% | **85.29%** | **0.3667** | **0.8054** | **Production Champion**: Catches 174 of 204 actual churners. |

---

##  Key Churn Drivers (Feature Importance)

Analysis of the Random Forest feature importances reveals the top 4 operational drivers of churn:

1. **`csat_score`**: Customer satisfaction score is the single strongest indicator. A score drop below `3.0` indicates immediate churn risk.
2. **`tenure_months`**: Risk is heavily concentrated in early customer lifecycles (first 90–180 days).
3. **`monthly_logins`**: A decline in login frequency acts as a precursor signal before cancellation.
4. **`payment_failures`**: Involuntary churn caused by expired cards or billing friction.

---

##  Repository Structure

```text
├── data/
│   └── customer_churn_business_dataset.csv  # Raw dataset (~10k records)
├── models/
│   ├── rf_model.joblib                      # Serialized Random Forest classifier
│   └── model_features.joblib                # Production feature schema (47 encoded columns)
├── api.py                                   # Production REST API with FastAPI
├── streamlit_app.py                         # Interactive Web Dashboard with Streamlit
├── train_model.py                           # Standalone training & artifact export script
├── data_analysis.ipynb                      # 6-stage ML notebook (EDA, baseline, evaluation)
├── ML_WORKFLOW.md                           # Comprehensive documentation of the 6 ML stages
├── requirements.txt                         # Cloud-ready Python dependencies
└── README.md                                # Project documentation
```

---

##  Quick Start

### 1. Try the Live Web App (No Installation Needed)
Access the live deployment directly in your browser:
 **[https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/](https://churn-prediction-95i4j7he3mtrzicna69ad5.streamlit.app/)**

---

### 2. Local Environment Setup

Clone the repository and install dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/maxr456/Churn-Prediction.git
cd Churn-Prediction

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### 3. Run the Web Dashboard Locally (`Streamlit`)

```bash
streamlit run streamlit_app.py
```

* Local URL: **`http://localhost:8501`**

---

### 4. Run the Production REST API (`FastAPI`)

The FastAPI service exposes a low-latency endpoint for programmatic integration with web platforms, CRMs, or mobile backends.

```bash
uvicorn api:app --reload --port 8000
```

* Local URL: **`http://localhost:8000`**
* Interactive Swagger Docs: **`http://localhost:8000/docs`**

#### Sample API Request (`POST /predict`):
```json
{
  "tenure_months": 6,
  "monthly_logins": 5,
  "csat_score": 2.1,
  "payment_failures": 2,
  "contract_type": "Monthly",
  "monthly_fee": 49
}
```

#### Sample API Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.58,
  "risk_level": "High",
  "threshold_used": 0.20,
  "recommended_action": "Immediate retention action required: CSAT review & discount offer."
}
```

---

### 5. Retrain Model from Scratch

To retrain the Random Forest model and update the serialized artifacts in `models/`:

```bash
python train_model.py
```

---

## 📖 Methodology & Deep-Dive

For a detailed breakdown of each machine learning stage (EDA, handling missing data, dummy variable trap, data leakage prevention, and metric derivations), refer to [ML_WORKFLOW.md](ML_WORKFLOW.md).
