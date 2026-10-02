from pathlib import Path
import pandas as pd
from src.preprocessing.cleaning import (
    load_data,
    clean_column_names,
    handle_missing_and_sentinels,
    drop_duplicates_and_invalid,
    encode_target,
)

from src.preprocessing.transforms import impute_missing_values

from src.preprocessing.feature_engineering import (
    run_feature_engineering
)

# Run cleaning pipeline
def run_preprocessing_pipeline(raw_data_path: str | Path) -> pd.DataFrame:
    # Cleaning steps
    df = load_data(raw_data_path)
    df = clean_column_names(df)
    df = handle_missing_and_sentinels(df)
    df = drop_duplicates_and_invalid(df)
    df = encode_target(df, target_col="churn")
    
    # Added in case the database has missing values
    # Transformation steps
    df = impute_missing_values(df) # KNN
    df = run_feature_engineering(df)
    
    return df