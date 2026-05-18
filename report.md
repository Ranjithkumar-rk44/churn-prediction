# Week 1 Report — Customer Churn Prediction

## Dataset
- Source: Telco Customer Churn (Kaggle)
- Total customers: 7043
- Total columns: 21

## Key Findings
- Overall churn rate: 26.5%
- Month-to-month contracts: 42% churn rate
- Two-year contracts: only 3% churn rate
- Churned customers pay $74.44 per month
- Loyal customers pay $61.27 per month
- Churned customers leave within 18 months
- Loyal customers stay for 37 months

## Data Cleaning Done
- Fixed TotalCharges column from text to number
- Filled 11 missing values using median
- Final shape: 7043 rows, 21 columns
- Zero missing values after cleaning

## Charts Created
- Churn count bar chart
- Churn by contract type
- Churn by tenure histogram
- Monthly charges boxplot
- Tenure boxplot
- Churn by internet service
- Correlation heatmap

## Next Steps — Week 2
- Feature engineering
- Train ML models
- Compare Logistic Regression, Random Forest, XGBoost