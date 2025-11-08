# Credit Default Prediction using Azure Machine Learning & MLflow  + Local Demo


# Overview

This project predicts whether a credit card client will default on their next payment, using machine learning models built and deployed on Azure Machine Learning (AML).

It showcases a full MLOps lifecycle — from data ingestion and training, to model registration, evaluation, and deployment as a live web endpoint.

To make the project richer and comparable, we train and evaluate three models:

1. Gradient Boosting Classifier (ensemble)
2. Logistic Regression (baseline)
3. XGBoost


# Objectives

- Build and train ML models using Azure Machine Learning SDK.

- Log experiments automatically with MLflow.

- Compare baseline and advanced models.

- Register the best model to the AML Workspace.

- Deploy the model as a Managed Online Endpoint.

- Test predictions via REST API.

**What:** End-to-end ML project predicting credit card default using Logistic Regression, Gradient Boosting, and XGBoost.  
**Stack:** Python, scikit-learn, XGBoost, MLflow, Azure ML SDK (v2), Streamlit.

## Features
- Train models locally or submit as Azure ML `command` jobs.
- MLflow auto-logging + lightweight local metrics CSV summarization.
- Optional model registration to Azure ML (controlled by CLI flags).
- Deploy endpoints on Azure ML
- Streamlit demo to run predictions locally / on free hosting platforms.

## Quick start (local)
```bash
git clone <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train gradient boosting locally
python src/train_gb.py --data_path data/default_of_credit_card_clients.csv --output_dir outputs/gb_run

# Run demo
streamlit run app/app.py -- --model-path outputs/gb_run/model.pkl



| Setting             | Choice                                          |
| ------------------- | ----------------------------------------------- |
| **Repo Name**       | `credit-default-aml`                            |
| **Deployment**      | Azure ML + Local Streamlit/FastAPI demo         |
| **Models Included** | Logistic Regression, Gradient Boosting, XGBoost |
| **Environment**     | `requirements.txt` (pip-based)                  |
| **Visualization**   | MLflow + Matplotlib metric comparison           |
| **Author Credit**   | © 2025 Ekpenyong Okpo                           |
