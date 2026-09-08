
from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC = [
    "Age","Account_Age_Months","Monthly_Charges","Total_Charges",
    "Monthly_Usage_GB","Number_of_Sessions","Support_Calls",
    "Complaints","Payment_Delay_Rate","Usage_Change_Percentage"
]
CATEGORICAL = [
    "Gender","Contract_Type","Internet_Service","Phone_Service",
    "Streaming_Service","Payment_Method"
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    # Business-friendly outlier clipping
    for col in ["Monthly_Charges","Total_Charges","Monthly_Usage_GB","Number_of_Sessions","Support_Calls","Complaints"]:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)
    return df

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Customer_Lifetime_Months"] = df["Account_Age_Months"]
    df["Average_Monthly_Spend"] = df["Total_Charges"] / df["Account_Age_Months"].clip(lower=1)
    df["Support_Calls_Per_Month"] = df["Support_Calls"] / df["Account_Age_Months"].clip(lower=1)
    df["Service_Count"] = (
        (df["Phone_Service"]=="Yes").astype(int)
        + (df["Streaming_Service"]=="Yes").astype(int)
        + (df["Internet_Service"]!="None").astype(int)
    )
    if "Payment_Delay_Rate" not in df:
        df["Payment_Delay_Rate"] = 0
    if "Usage_Change_Percentage" not in df:
        df["Usage_Change_Percentage"] = 0
    return df
