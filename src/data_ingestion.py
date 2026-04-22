import pandas as pd
import logging
import os
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(source_type: str = "db", file_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Loads data either from the MySQL database or fallback CSV.
    """
    try:
        if source_type == "db":
            logging.info("Attempting to load data from MySQL Database...")
            load_dotenv()
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "3306")
            db_user = os.getenv("DB_USER", "root")
            db_password = os.getenv("DB_PASSWORD", "")
            db_name = os.getenv("DB_NAME", "telco_churn_db")
            
            if not db_password:
                raise ValueError("Database password is not set in .env")
                
            encoded_password = urllib.parse.quote_plus(db_password)
            engine = create_engine(f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}")
            query = "SELECT * FROM customers"
            df = pd.read_sql(query, engine)
            logging.info(f"Data successfully loaded from MySQL. Shape: {df.shape}")
            return df
            
        else:
            logging.info(f"Attempting to load data from {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
                
            df = pd.read_csv(file_path)
            logging.info(f"Data successfully loaded from CSV. Shape: {df.shape}")
            return df
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        logging.info("Falling back to CSV...")
        if source_type == "db":
            return load_data(source_type="csv", file_path=file_path)
        raise

if __name__ == "__main__":
    # Test the function
    df = load_data(source_type="csv")
    print(df.head())
