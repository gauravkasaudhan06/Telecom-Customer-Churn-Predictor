from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import logging
import sys
import os

# Add the current directory to path so imports work seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict_churn

# Initialize the FastAPI app
app = FastAPI(
    title="Telco Customer Churn API",
    description="An API to predict the probability of a customer churning, with SHAP explanations.",
    version="1.0.0"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected data schema using Pydantic
class CustomerData(BaseModel):
    SeniorCitizen: int = Field(..., example=0)
    tenure: int = Field(..., example=1)
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    TechSupport: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: str = Field(..., example="29.85")

@app.get("/")
def home():
    return {"message": "Welcome to the Telco Customer Churn Prediction API. Go to /docs for the interactive API documentation."}

@app.post("/predict")
def predict_endpoint(customer: CustomerData):
    """
    Predicts the churn probability for a single customer.
    """
    try:
        # Convert the Pydantic object to a pandas DataFrame
        # model_dump() replaces dict() in Pydantic v2
        customer_dict = customer.model_dump()
        
        # We need to pass a DataFrame with a single row
        df_input = pd.DataFrame([customer_dict])
        
        # Run inference using our pre-built pipeline
        # Assuming the model files are in the root directory relative to where the server is run
        results = predict_churn(
            df_input, 
            model_path="xgboost_model.pkl", 
            scaler_path="scaler.pkl", 
            feature_cols_path="feature_columns.pkl"
        )
        
        # The result is a list of dicts, we just want the first one
        return results[0]
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
