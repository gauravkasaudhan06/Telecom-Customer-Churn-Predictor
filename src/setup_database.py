import pandas as pd
import os
import logging
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mysql_database(csv_path: str):
    """
    Reads CSV and uploads it to MySQL database using credentials from .env.
    """
    load_dotenv()
    
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "telco_churn_db")
    
    if not db_password:
        logging.error("Database password is empty! Please fill it in the .env file.")
        return
        
    encoded_password = urllib.parse.quote_plus(db_password)
    
    # 1. Connect to MySQL Server (without specifying a database) to create the DB if it doesn't exist
    server_engine = create_engine(f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/")
    try:
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            logging.info(f"Database '{db_name}' ensured to exist.")
    except Exception as e:
        logging.error(f"Failed to connect to MySQL server. Please ensure MySQL is running and credentials are correct. Error: {e}")
        return

    # 2. Connect to the specific database
    db_engine = create_engine(f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}")
    
    # 3. Read CSV and upload
    try:
        logging.info(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        logging.info("Uploading data to MySQL table 'customers'...")
        df.to_sql(name="customers", con=db_engine, if_exists="replace", index=False)
        logging.info(f"Successfully uploaded {len(df)} records to the database!")
        
    except Exception as e:
        logging.error(f"Failed to upload data: {e}")

if __name__ == "__main__":
    setup_mysql_database("WA_Fn-UseC_-Telco-Customer-Churn.csv")
