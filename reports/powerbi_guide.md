# Power BI Implementation Guide

The case study lists Power BI as a business-intelligence technology. A `.pbix` file cannot be generated reliably outside Power BI Desktop, so this project includes the exact model/dashboard plan.

## Import
Load `data/processed/customer_churn_processed.csv`.

## Suggested pages
1. Executive Overview
2. Customer Segmentation
3. Churn Drivers
4. Retention Priorities

## KPI cards
- Total Customers
- Churn Rate
- High-Risk Customers
- Estimated Revenue at Risk

## Recommended visuals
- Churn by Contract Type
- Churn by Tenure Band
- Monthly Charges vs Churn
- Payment Method vs Churn
- Support Calls vs Churn
- Feature Importance
- High-risk customer table

## Core DAX examples
```DAX
Total Customers = DISTINCTCOUNT(Customer[Customer_ID])
Churned Customers = CALCULATE([Total Customers], Customer[Churn] = "Yes")
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
Revenue at Risk = SUMX(Customer, Customer[Monthly_Charges] * 12 * Customer[Churn_Probability])
```
