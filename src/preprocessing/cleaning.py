from pathlib import Path
import pandas as pd
import numpy as np

# Load data function from path
def load_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file provided {path} is not found!")
    return pd.read_csv(path)

# Clean column names in dataframe
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    new_cols = []
    for col in df_clean.columns:
        if col.lower() == "customerid":
            new_cols.append("customer_id")
        else:
            snake = "".join(["_" + c.lower() if c.isupper() else c for c in col]).lstrip("_")
            new_cols.append(snake)
    df_clean.columns = new_cols
    return df_clean

# Handle missing and sentinel rows
def handle_missing_and_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.replace(r"^\s*$", np.nan, regex=True)
    
    if "total_charges" in df_clean.columns:
        df_clean["total_charges"] = pd.to_numeric(df_clean["total_charges"], errors="coerce")
        
    if "tenure" in df_clean.columns and "total_charges" in df_clean.columns:
        df_clean.loc[df_clean["tenure"] == 0, "total_charges"] = 0.0
        
    return df_clean

# Drop duplicates and invalid rows
def drop_duplicates_and_invalid(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    initial_shape = df_clean.shape
    df_clean = df_clean.drop_duplicates()
    
    print(f"Removed Duplicates {initial_shape[0] - df_clean.shape[0]}")
    
    return df_clean

# Encode specific column
def encode_target(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    df_clean = df.copy()
    if target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].replace(
            {"Yes": 1, "No": 0, "yes": 1, "no": 0, "YES": 1, "NO": 0}
        )
    return df_clean

