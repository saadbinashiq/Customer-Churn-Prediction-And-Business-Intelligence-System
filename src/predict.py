
from pathlib import Path
import json, pickle
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT/"models/churn_model.pkl"

RISK_THRESHOLDS = {"HIGH": 0.65, "MEDIUM": 0.40}

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def risk_level(p):
    if p >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if p >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"

def explain_customer(row):
    reasons = []
    if row.get("Account_Age_Months", 99) < 12:
        reasons.append(("Short customer tenure", "High"))
    if row.get("Monthly_Charges", 0) > 90:
        reasons.append(("High monthly charges", "High"))
    if row.get("Support_Calls", 0) >= 4:
        reasons.append(("Frequent support requests", "High"))
    if row.get("Contract_Type","") == "Month-to-month":
        reasons.append(("Month-to-month contract", "High"))
    if row.get("Payment_Delay_Rate", 0) > 0.25:
        reasons.append(("High payment delay rate", "Medium"))
    if row.get("Complaints", 0) >= 2:
        reasons.append(("Multiple complaints", "Medium"))
    if row.get("Usage_Change_Percentage", 0) < -15:
        reasons.append(("Declining usage", "Medium"))
    if not reasons:
        reasons.append(("No dominant rule-based risk driver detected", "Low"))
    return reasons[:5]

def recommendation(level, reasons):
    if level == "HIGH":
        return "Prioritize proactive retention: offer a contract upgrade/discount, assign support follow-up, and review pricing."
    if level == "MEDIUM":
        return "Monitor closely and offer a targeted service or billing intervention before the next renewal."
    return "Maintain engagement with regular service value messaging and loyalty offers."

def predict(df):
    model = load_model()
    X = df.drop(columns=[c for c in ["Customer_ID","Churn"] if c in df.columns])
    probs = model.predict_proba(X)[:,1]
    out = df.copy()
    out["Churn_Probability"] = probs
    out["Risk_Level"] = [risk_level(p) for p in probs]
    out["Estimated_Revenue_At_Risk"] = out["Monthly_Charges"] * 12 * probs
    return out
