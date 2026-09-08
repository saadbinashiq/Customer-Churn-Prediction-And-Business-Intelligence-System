
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT/"models/model_metrics.csv")
print("\nMODEL COMPARISON\n")
print(m.to_string(index=False))
print("\nBest by ROC-AUC:", m.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"])
