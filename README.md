# 🌟 Telecom Customer Retention Hub

An end-to-end Machine Learning pipeline and Business Intelligence dashboard designed to predict telecom customer churn and provide actionable retention strategies. 

## 🚀 Features
* **Predictive AI Model:** Uses **XGBoost** to accurately predict the likelihood of a customer leaving based on historical demographics, account, and service data.
* **Explainable AI (XAI):** Integrates **SHAP** to demystify the AI's decision-making process, providing context-aware, human-readable "Key Reasons" for high churn risk instead of raw numerical outputs.
* **Interactive Dashboard:** Built with **Streamlit**, featuring a modern, PowerBI-style UI with custom CSS, Lottie animations, and dynamic business metrics (e.g., tracking "Lost MRR" to quantify business impact).
* **Data Engineering:** Modular Python scripts to handle secure data ingestion from a MySQL database, with a built-in fallback to CSV for cloud environments.
* **MLOps Tracking:** Uses **MLflow** to track experiments, hyperparameters, and model accuracy during the training phase.

## 📂 Project Structure
* `src/app.py`: The main Streamlit Dashboard frontend application.
* `src/train.py`: The machine learning training pipeline (XGBoost + SMOTE + MLflow).
* `src/predict.py`: Inference engine and SHAP value calculator.
* `src/data_ingestion.py`: Secure database connection and data loading module.
* `src/preprocessing.py`: Data cleaning, encoding, and feature selection module.
* `requirements.txt`: List of all Python dependencies.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd customer_churn_prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   ```bash
   streamlit run src/app.py
   ```

## ☁️ Deployment (Streamlit Community Cloud)
This project is deployment-ready for Streamlit Community Cloud. The `data_ingestion.py` module contains a smart fallback mechanism to read directly from the dataset `.csv` file if the local MySQL database is unreachable, ensuring a seamless, zero-config cloud execution. Simply link this repository to your Streamlit Cloud account and set `src/app.py` as the main file path.
