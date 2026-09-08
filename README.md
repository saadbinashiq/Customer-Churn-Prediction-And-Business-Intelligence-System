# Customer Churn Prediction & Business Intelligence System

A complete portfolio-grade Data Science / Machine Learning / Business Analytics project based directly on the supplied case study.

## What is included
- Data cleaning: missing values, duplicates, outlier treatment, consistency handling
- EDA: churn distribution, tenure, monthly charges, contracts, support, payment methods
- Business feature engineering:
  - Customer_Lifetime_Months
  - Average_Monthly_Spend
  - Support_Calls_Per_Month
  - Payment_Delay_Rate
  - Service_Count
  - Usage_Change_Percentage
- Model comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost (when installed)
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Explainability: permutation importance + business-rule explanations
- Interactive dark responsive Streamlit dashboard
- Individual/batch churn prediction and CSV download
- Revenue-at-risk estimation
- SQL/PostgreSQL schema
- Jupyter notebooks for the requested workflow
- Business report PDF
- Power BI implementation guide
- GitHub/deployment guidance

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python src/train_model.py
streamlit run dashboard/app.py
```

Open the Streamlit URL shown in the terminal.

## Important
The supplied case study explicitly says model metric values should come from actual experiments rather than being invented. Therefore this project generates a reproducible synthetic customer dataset and trains the models locally; `models/model_metrics.csv` contains the actual results from that training run.

If you have a real telecom/SaaS/e-commerce dataset with the same fields, replace `data/raw/customer_churn.csv` and retrain.

## Deployment
Recommended simple path:
1. Push this folder to GitHub.
2. Deploy the repository on Streamlit Community Cloud.
3. Main file: `dashboard/app.py`.
4. Python dependencies are in `requirements.txt`.

For PostgreSQL, see `sql/schema.sql`.
For Power BI, see `reports/powerbi_guide.md`.
