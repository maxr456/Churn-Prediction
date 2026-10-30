# Customer Churn Prediction

Predicting customer churn using business metrics, user engagement patterns, and machine learning models.

## Project Structure
- [`streamlit_app.py`](streamlit_app.py): Interactive web dashboard built with Streamlit.
- [`api.py`](api.py): Production REST API built with FastAPI.
- [`train_model.py`](train_model.py): Standalone script to train the model and export production artifacts to `models/`.
- [`models/`](models/): Serialized Random Forest model and feature definitions.
- [`data_analysis.ipynb`](data_analysis.ipynb): Jupyter notebook containing end-to-end EDA, data cleaning, baseline modeling, and metric comparisons.
- [`ML_WORKFLOW.md`](ML_WORKFLOW.md): Step-by-step documentation detailing the 6 stages of the machine learning lifecycle for this project.

## Quick Start

### 1. Setup Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

### 3. Run the Production REST API
```bash
uvicorn api:app --reload --port 8000
```
Interactive API documentation available at `http://localhost:8000/docs`.
