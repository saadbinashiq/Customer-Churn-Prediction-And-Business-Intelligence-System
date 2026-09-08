
from pathlib import Path
import json, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from data_preprocessing import clean_data, add_business_features

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT/"data/raw/customer_churn.csv"
model_dir = ROOT/"models"
processed_dir = ROOT/"data/processed"
model_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
df = add_business_features(clean_data(df))
df.to_csv(processed_dir/"customer_churn_processed.csv", index=False)

y = (df["Churn"]=="Yes").astype(int)
X = df.drop(columns=["Customer_ID","Churn"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=260, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, eval_metric="logloss", random_state=42
    )
except Exception:
    pass

results = []
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    p = pipe.predict_proba(X_test)[:,1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test,pred),
        "Precision": precision_score(y_test,pred,zero_division=0),
        "Recall": recall_score(y_test,pred,zero_division=0),
        "F1": f1_score(y_test,pred,zero_division=0),
        "ROC_AUC": roc_auc_score(y_test,p),
        "PR_AUC": average_precision_score(y_test,p)
    })
    fitted[name] = pipe

metrics = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
metrics.to_csv(model_dir/"model_metrics.csv", index=False)

best_name = metrics.iloc[0]["Model"]
best = fitted[best_name]
with open(model_dir/"churn_model.pkl","wb") as f:
    pickle.dump(best,f)

metadata = {
    "best_model": best_name,
    "features": X.columns.tolist(),
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "churn_rate": float(y.mean())
}
(model_dir/"model_metadata.json").write_text(json.dumps(metadata, indent=2))

# Permutation importance on original feature matrix using best pipeline
perm = permutation_importance(best, X_test, y_test, n_repeats=4, random_state=42, scoring="roc_auc")
imp = pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean})
imp = imp.sort_values("Importance", ascending=False)
imp.to_csv(model_dir/"feature_importance.csv", index=False)

print(metrics.to_string(index=False))
print(f"\nBest model: {best_name}")
