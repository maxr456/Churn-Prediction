"""
train_model.py
Trains the Customer Churn Random Forest model and saves the artifacts for production.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_export():
    print("Loading dataset...")
    data_path = os.path.join("data", "customer_churn_business_dataset.csv")
    df = pd.read_csv(data_path)

    # 1. Clean data and handle missing values
    df_clean = df.drop(columns=["customer_id"]).fillna({"complaint_type": "No Complaint"})
    X = df_clean.drop(columns=["churn"])
    y = df_clean["churn"]

    # 2. One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
    feature_columns = list(X_encoded.columns)

    # 3. Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Train Random Forest with balanced class weights
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)

    # 5. Export artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, os.path.join("models", "rf_model.joblib"))
    joblib.dump(feature_columns, os.path.join("models", "model_features.joblib"))

    print(f"Artifacts successfully saved in 'models/' directory! ({len(feature_columns)} features)")

if __name__ == "__main__":
    train_and_export()

