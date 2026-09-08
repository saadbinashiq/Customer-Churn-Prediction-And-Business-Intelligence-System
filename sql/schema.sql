
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(30) PRIMARY KEY,
    age INT,
    gender VARCHAR(20),
    account_age_months INT,
    contract_type VARCHAR(40),
    internet_service VARCHAR(40),
    phone_service VARCHAR(10),
    streaming_service VARCHAR(10),
    monthly_charges NUMERIC(12,2),
    total_charges NUMERIC(14,2),
    monthly_usage_gb NUMERIC(12,2),
    number_of_sessions INT,
    support_calls INT,
    complaints INT,
    payment_method VARCHAR(40),
    payment_delay_rate NUMERIC(8,4),
    usage_change_percentage NUMERIC(10,2),
    churn VARCHAR(5)
);

CREATE OR REPLACE VIEW churn_business_summary AS
SELECT
    COUNT(*) AS total_customers,
    ROUND(100.0 * AVG(CASE WHEN churn='Yes' THEN 1 ELSE 0 END), 2) AS churn_rate,
    ROUND(SUM(CASE WHEN churn='Yes' THEN monthly_charges*12 ELSE 0 END), 2) AS annual_revenue_at_risk
FROM customers;
