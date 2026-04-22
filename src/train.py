import os
import pickle
import logging
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Local imports
from data_ingestion import load_data
from preprocessing import clean_data, preprocess_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path: str, model_save_path: str = "xgboost_model.pkl"):
    """
    Trains an XGBoost model on the dataset, tracks experiments with MLflow, and saves it.
    """
    logging.info("Starting model training pipeline with MLflow tracking...")
    
    # Set MLflow experiment
    mlflow.set_experiment("Telco_Customer_Churn")
    
    # 1. Ingest Data
    df = load_data(data_path)
    
    # 2. Clean Data
    df_cleaned = clean_data(df)
    
    # 3. Preprocess Features (Scaling and Encoding)
    df_preprocessed = preprocess_features(df_cleaned, is_training=True, scaler_path="scaler.pkl")
    
    # 4. Split Features and Target
    X = df_preprocessed.drop(columns=["Churn"])
    y = df_preprocessed["Churn"]
    
    # Save the expected feature names
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Start MLflow run
    with mlflow.start_run():
        logging.info("Training XGBoost Classifier...")
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("smote_applied", True)
        
        # 7. Train XGBoost Model
        model_xgb = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model_xgb.fit(X_train_smote, y_train_smote)
        
        # 8. Evaluate Model
        y_pred = model_xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logging.info(f"Model Accuracy: {accuracy:.4f}")
        logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model_xgb, "model")
        
        # 9. Save Model locally
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_xgb, f)
        logging.info(f"Model saved locally to {model_save_path}")

if __name__ == "__main__":
    data_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    train_model(data_file)
