import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Encode categorical values
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    columns_to_encode = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if "customer_id" in columns_to_encode:
        columns_to_encode.remove("customer_id")
        
    if columns_to_encode:
        df_clean = pd.get_dummies(df_clean, columns=columns_to_encode, drop_first=True, dtype=int)
        
    return df_clean

# Scale numerical values
def scale_numerical(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    # Standard numerical columns 
    numerical_cols = [str(col) for col in ["tenure", "monthly_charges", "total_charges"]]
    cols_to_scale = [col for col in numerical_cols if col in df_clean.columns]
    
    if cols_to_scale:
        scaler = StandardScaler()
        df_clean[cols_to_scale] =  scaler.fit_transform(df_clean[cols_to_scale])
        
    return df_clean


# Added in case the database has missing values!
def impute_missing_values(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df_clean = df.copy()
    
    num_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
    
    if num_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
    
    return df_clean 