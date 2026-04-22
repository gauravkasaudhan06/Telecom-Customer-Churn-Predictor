import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values, standardizing categorical strings,
    and converting types.
    """
    logging.info("Starting data cleaning...")
    df = df.copy()
    
    # Drop columns that have very low feature importance to optimize the model and UI
    useless_cols = ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 
                    'MultipleLines', 'DeviceProtection', 'StreamingTV', 'StreamingMovies', 
                    'PaperlessBilling']
    df = df.drop(columns=[col for col in useless_cols if col in df.columns], errors='ignore')
    
    # Handle TotalCharges which is read as object because of empty spaces
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing values with 0 (since they represent new customers with 0 tenure)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Drop rows with missing values in other columns if any
    df = df.dropna()
    
    # 3. Standardize 'No internet service' and 'No phone service' to 'No'
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'TechSupport']
    
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No'})
            
    # 4. Convert Yes/No to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
                      
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
            
    # 5. Convert gender to 1/0 (Female: 1, Male: 0)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
        
    logging.info("Data cleaning complete.")
    return df

def preprocess_features(df: pd.DataFrame, is_training: bool = True, scaler_path: str = "scaler.pkl") -> pd.DataFrame:
    """
    Applies one-hot encoding and min-max scaling to the features.
    Saves the scaler during training, and loads it during inference.
    """
    logging.info("Starting feature preprocessing...")
    df = df.copy()
    
    # 1. One-hot encoding for remaining categorical variables
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    # Filter to only encode columns that exist
    encode_cols = [c for c in categorical_cols if c in df.columns]
    
    df = pd.get_dummies(data=df, columns=encode_cols, dtype=int)
    
    # Ensure all expected columns exist (important for inference)
    # We will define the expected columns explicitly so that during prediction, 
    # if a category was missing in the input, the column is still created with 0s.
    expected_dummies = [
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = 0
            
    # 2. Min-Max Scaling
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    if is_training:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        # Save the scaler for later use
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler trained and saved to {scaler_path}")
    else:
        # Load the saved scaler
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            logging.info(f"Scaler loaded from {scaler_path}")
        except FileNotFoundError:
            logging.error(f"Scaler not found at {scaler_path}. Please train the model first.")
            raise
            
    logging.info("Feature preprocessing complete.")
    return df
