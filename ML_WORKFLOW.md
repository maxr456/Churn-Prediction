# End-to-End Churn Prediction: Project Workflow & 6 ML Stages

A step-by-step reference guide for building, evaluating, and interpreting the Customer Churn Prediction model.

---

## Project Overview
* **Dataset**: `customer_churn_business_dataset.csv` (~10,000 records, 32 features)
* **Target Variable**: `churn` (Binary classification: `1` = Churned, `0` = Retained)
* **Key Challenge**: Class imbalance (~10.2% churned vs ~89.8% retained)

---

## The 6 Stages of Machine Learning

```
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. Data Understanding & Inspection                            │
  │    Check shapes, data types, null counts, and identifiers     │
  └───────────────────────────────┬───────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 2. Exploratory Data Analysis (EDA)                            │
  │    Target balance, univariate/bivariate patterns, drivers     │
  └───────────────────────────────┬───────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 3. Data Cleaning & Feature Preprocessing                      │
  │    Drop IDs, handle nulls, encode categoricals, scale numbers │
  └───────────────────────────────┬───────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 4. Train / Test Split (Stratified)                            │
  │    Preserve 10% churn ratio; isolate test set from train set  │
  └───────────────────────────────┬───────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 5. Model Training & Comparison                                │
  │    Dummy Baseline -> Logistic Regression -> Random Forest     │
  └───────────────────────────────┬───────────────────────────────┘
                                  ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 6. Model Evaluation & Business Insights                       │
  │    Precision, Recall, F1, ROC-AUC, Feature Importance         │
  └───────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Data Understanding & Initial Inspection
**Objective**: Understand data structure, column definitions, and basic integrity.

* **Key Tasks**:
  1. Inspect shape (`df.shape`) and preview sample rows (`df.head()`).
  2. Identify data types (`df.info()`) — differentiate numerical columns from text/categorical columns.
  3. Identify missing values (`df.isnull().sum()`).
     * *Project specific*: `complaint_type` has missing values because non-complaining customers don't have a category.
  4. Flag non-predictive columns:
     * Identifiers like `customer_id` must not be fed to the model (they introduce memorization, not learning).

---

### Stage 2: Exploratory Data Analysis (EDA)
**Objective**: Uncover patterns, distributions, and initial correlations with churn.

* **Key Tasks**:
  1. **Analyze Target Distribution**:
     * Check `df['churn'].value_counts(normalize=True)`.
     * *Finding*: ~10.2% churn rate (severe imbalance).
  2. **Numerical Features vs. Churn**:
     * Compare means/medians of churned vs. retained users (e.g., `tenure_months`, `support_tickets`, `nps_score`, `last_login_days_ago`, `monthly_logins`).
  3. **Categorical Features vs. Churn**:
     * Compare churn rates across categories (e.g., `contract_type`, `payment_method`, `customer_segment`).
  4. **Hypothesis Generation**: Which factors seem to trigger churn? (e.g., price increases, recent inactivity, unresolved tickets).

---

### Stage 3: Data Cleaning & Feature Preprocessing
**Objective**: Prepare data into a numeric matrix suitable for machine learning models.

* **Key Tasks**:
  1. **Drop Identifiers**: Remove `customer_id`.
  2. **Handle Missing Values**:
     * For `complaint_type`, fill missing values with `"No Complaint"` rather than dropping rows.
  3. **Categorical Encoding**:
     * **Nominal** (no natural order like `country`, `payment_method`): One-Hot Encoding (`pd.get_dummies` or `OneHotEncoder`).
     * **Binary / Flags** (`discount_applied`, `price_increase_last_3m`): Map to `1`/`0`.
  4. **Feature Scaling (When needed)**:
     * Apply `StandardScaler` or `MinMaxScaler` for distance-based or linear models (Logistic Regression, KNN). Tree models (Random Forest, XGBoost) do not require scaling.

---

### Stage 4: Train / Test Split & Imbalance Strategy
**Objective**: Create an unbiased validation strategy that reflects real-world performance.

* **Key Tasks**:
  1. **Stratified Split**:
     * Use `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.
     * *Why?* Ensures both training (80%) and testing (20%) sets contain the exact same ~10.2% churn proportion.
  2. **Prevent Data Leakage**:
     * Fit scalers/encoders strictly on the **training data**, then transform the **test data**.
  3. **Address Imbalance**:
     * Use model parameters like `class_weight='balanced'`.
     * Optional advanced techniques: SMOTE (oversampling minority class) or downsampling.

---

### Stage 5: Model Training & Comparison
**Objective**: Train a succession of models from simple baselines to competitive ensemble models.

* **Recommended Sequence**:
  1. **Dummy / Naive Baseline**:
     * Always predicts the majority class (`0`). Gives ~89.8% accuracy but 0% recall for churn.
  2. **Linear Baseline (Logistic Regression)**:
     * Fast, interpretable, establishes linear separability.
  3. **Tree-Based Ensembles (Random Forest / LightGBM / XGBoost)**:
     * Handles non-linear relationships, feature interactions, and unscaled data well.

---

### Stage 6: Model Evaluation & Business Insights
**Objective**: Quantify model performance correctly and translate predictions into actionable retention actions.

* **Key Evaluation Metrics**:
  * ❌ **Accuracy**: Misleading! (A model predicting "no churn" for everyone is 90% accurate but useless).
  * ✅ **Precision**: When we predict a customer will churn, how often are we right? (Minimizes wasted retention offers).
  * ✅ **Recall**: What percentage of all actual churners did we catch? (Crucial if losing a customer is very costly).
  * ✅ **F1-Score**: Harmonic mean of Precision and Recall.
  * ✅ **ROC-AUC & PR-AUC**: Measures the model's ranking ability across all probability thresholds.
* **Business Insights (Explainability)**:
  * Extract **Feature Importance** (e.g., `model.feature_importances_` or SHAP values).
  * Determine the top 3-5 churn risk indicators so the business can intervene early.

---

## Golden Rules to Avoid Common Pitfalls

| Pitfall | Why It Is Dangerous | Correct Practice |
| :--- | :--- | :--- |
| **Data Leakage** | Preprocessing (e.g., scaling or imputing) on the entire dataset before splitting causes overly optimistic test scores. | Split into train/test **first**, then fit preprocessors only on the training set. |
| **Relying on Accuracy** | With 10% churn, 90% accuracy is the trivial baseline score. | Prioritize **Recall**, **Precision**, and **ROC-AUC / PR-AUC**. |
| **Keeping Customer ID** | Model can memorize ID strings or IDs correlated with order of collection. | Drop `customer_id` before model training. |
| **Unstratified Split** | Test set might end up with too few or too many churned users. | Always use `stratify=y` in `train_test_split`. |
