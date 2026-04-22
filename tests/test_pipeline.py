import pytest
import pandas as pd
import sys
import os

# Add src to the python path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_ingestion import load_data
from preprocessing import clean_data

def test_load_data_file_not_found():
    """Test if load_data raises an error for non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_data("fake_file.csv")

def test_clean_data():
    """Test if clean_data correctly formats columns and drops customerID."""
    # Create dummy dataframe mimicking the raw data
    dummy_data = pd.DataFrame({
        'customerID': ['1234-ABCD', '5678-EFGH'],
        'TotalCharges': ['29.85', ' '],
        'Partner': ['Yes', 'No'],
        'gender': ['Female', 'Male']
    })
    
    cleaned_df = clean_data(dummy_data)
    
    # 1. Check customerID is dropped
    assert 'customerID' not in cleaned_df.columns
    
    # 2. Check TotalCharges empty space is replaced by 0.0 and converted to float
    assert cleaned_df['TotalCharges'].iloc[1] == 0.0
    assert cleaned_df['TotalCharges'].dtype == float
    
    # 3. Check Yes/No is converted to 1/0
    assert cleaned_df['Partner'].iloc[0] == 1
    assert cleaned_df['Partner'].iloc[1] == 0
    
    # 4. Check gender is converted Female:1, Male:0
    assert cleaned_df['gender'].iloc[0] == 1
    assert cleaned_df['gender'].iloc[1] == 0
