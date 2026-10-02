from pathlib import Path
import numpy as np 
import pandas as pd
import pytest 

from src.preprocessing.cleaning import (
    clean_column_names,
    handle_missing_and_sentinels,
    drop_duplicates_and_invalid,
    encode_target
)

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    data = {
      "customerID": ["1111-ABCDE", "2222-FGHIJ", "2222-FGHIJ", "3331-KLMNO"],
      "tenure": [1, 0, 0, 24],
      "MonthlyCharges": [29.85, 50.00, 50.00, 70.00],
      "TotalCharges": ["29.85", "   ", "   ", "1680.0"],
      "Churn": ["No", "Yes", "Yes", "No"],
    }
    return pd.DataFrame(data)

def test_clean_column_names(sample_raw_df):
    cleaned = clean_column_names(sample_raw_df)
    assert "customer_id" in cleaned.columns
    assert "monthly_charges" in cleaned.columns
    assert "total_charges" in cleaned.columns
    assert "churn" in cleaned.columns
    
def test_handle_missing_and_sentinels(sample_raw_df):
    cleaned = clean_column_names(sample_raw_df)
    cleaned = handle_missing_and_sentinels(cleaned)
    assert cleaned.loc[cleaned["tenure"] == 0, "total_charges"].values[0] == 0.0
    assert pd.api.types.is_float_dtype(cleaned["total_charges"])
    
def test_drop_duplicates_and_invalid(sample_raw_df):
    cleaned = clean_column_names(sample_raw_df)
    assert len(cleaned) == 4
    cleaned = drop_duplicates_and_invalid(cleaned)
    assert len(cleaned) == 3
    
def test_encode_target(sample_raw_df):
    cleaned = clean_column_names(sample_raw_df)
    cleaned = drop_duplicates_and_invalid(cleaned)
    cleaned = encode_target(cleaned, target_col = "churn")
    assert set(cleaned["churn"].unique()).issubset({0, 1})
    assert cleaned.loc[cleaned["customer_id"] == "2222-FGHIJ", "churn"].values[0] == 1