import pandas as pd
import numpy as np
import pickle
import logging
import shap
from preprocessing import clean_data, preprocess_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_top_reasons(shap_values, feature_names, top_n=3):
    """
    Extracts the top N positive contributing features for churn based on SHAP values.
    """
    # Create a dictionary of feature names to their SHAP value for this instance
    feature_contributions = dict(zip(feature_names, shap_values))
    # Filter for positive contributions (things driving churn probability UP)
    positive_contributions = {k: v for k, v in feature_contributions.items() if v > 0}
    # Sort by impact
    sorted_contributions = sorted(positive_contributions.items(), key=lambda item: item[1], reverse=True)
    
    top_reasons = [feat for feat, val in sorted_contributions[:top_n]]
    if not top_reasons:
        return ["No strong factors driving churn"]
    return top_reasons

def calculate_business_action(probability: float, monthly_charges: float) -> str:
    """
    Determines the business action based on churn probability and customer value.
    """
    if probability > 0.80:
        if monthly_charges > 70:
            return f"Action: High Value Risk. Offer 20% discount on a 1-year contract immediately. (Loss averted: ~${monthly_charges*12:.2f}/yr)"
        else:
            return "Action: Standard Risk. Offer 10% discount on next month."
    elif probability > 0.50:
        return "Action: Monitor. Send a customer satisfaction survey."
    else:
        return "Action: No intervention needed."

def predict_churn(input_data: pd.DataFrame, model_path: str = "xgboost_model.pkl", scaler_path: str = "scaler.pkl", feature_cols_path: str = "feature_columns.pkl"):
    """
    Predicts churn for new input data and generates SHAP explanations and business actions.
    """
    logging.info("Starting inference pipeline...")
    
    # 1. Clean Data
    df_cleaned = clean_data(input_data)
    original_monthly_charges = df_cleaned['MonthlyCharges'].values
    
    # 2. Preprocess Features (Scaling and Encoding using saved scaler)
    df_preprocessed = preprocess_features(df_cleaned, is_training=False, scaler_path=scaler_path)
    
    # 3. Ensure columns match the training data exactly
    with open(feature_cols_path, 'rb') as f:
        expected_columns = pickle.load(f)
        
    for col in expected_columns:
        if col not in df_preprocessed.columns:
            df_preprocessed[col] = 0
    df_final = df_preprocessed[expected_columns]
    
    # 4. Load XGBoost Model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 5. Predict
    predictions = model.predict(df_final)
    probabilities = model.predict_proba(df_final)[:, 1]
    
    # 6. SHAP Explanations
    explainer = shap.TreeExplainer(model)
    # SHAP values for XGBoost binary classification usually return a matrix
    shap_values = explainer.shap_values(df_final)
    
    results = []
    for i in range(len(predictions)):
        prob = probabilities[i]
        monthly_charge = original_monthly_charges[i]
        
        # Handle SHAP output shape depending on the explainer
        if isinstance(shap_values, list):
            sv = shap_values[1][i]
        else:
            sv = shap_values[i]
            
        top_reasons = get_top_reasons(sv, expected_columns)
        action = calculate_business_action(prob, monthly_charge)
        
        results.append({
            "Prediction": int(predictions[i]),
            "Probability": round(prob, 4),
            "Top_Churn_Drivers": top_reasons,
            "Recommended_Action": action
        })
    
    logging.info("Inference complete.")
    return results

if __name__ == "__main__":
    # Example usage with a dummy record
    dummy_data = {
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [90.85],
        'TotalCharges': ['90.85']
    }
    
    df_new = pd.DataFrame(dummy_data)
    try:
        results = predict_churn(df_new)
        print("\n--- Inference Results ---")
        for res in results:
            print(f"Prediction (1=Churn): {res['Prediction']}")
            print(f"Probability: {res['Probability']*100:.1f}%")
            print(f"Top Drivers for Churn: {res['Top_Churn_Drivers']}")
            print(res['Recommended_Action'])
    except Exception as e:
        logging.error(f"Prediction failed. Did you run train.py first? Error: {e}")
