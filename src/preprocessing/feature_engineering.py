import pandas as pd
import numpy as np

def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    if "tenure" in df_feat.columns:
        bins = [-1, 12, 24, 48, 72]
        labels = ["0-12_months", "13-24_months", "25-48_months", "49-72_months"]
        
        df_feat["tenure_group"] = pd.cut(df_feat["tenure"], bins=bins, labels=labels)
        
        return df_feat
    
    
def count_total_services(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    service_cols = [
        "phone_service", 
        "multiple_lines", 
        "online_security", 
        "online_backup", 
        "device_protection", 
        "tech_support", 
        "streaming_t_v", 
        "streaming_movies"
    ]
    
    existing_cols = [col for col in service_cols if col in df_feat.columns]
    
    if existing_cols:
        df_feat["total_services_count"] = (df_feat[existing_cols] == 'Yes').sum(axis=1)
    else: 
        df_feat["total_services_count"] = 0
        
    return df_feat

def calculate_spend_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy() 
    
    if "total_charges" in df_feat.columns and "tenure" in df_feat.columns and "monthly_charges" in df_feat.columns:
        expected_charges = df_feat["tenure"] * df_feat["monthly_charges"]
        
        expected_charges = np.where(expected_charges == 0, df_feat["monthly_charges"], expected_charges)
        
        df_feat["avg_monthly_spend_ratio"] = df_feat["total_charges"] / expected_charges    
    return df_feat

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = create_tenure_groups(df)
    df = count_total_services(df)
    df = calculate_spend_ratios(df)
    return df