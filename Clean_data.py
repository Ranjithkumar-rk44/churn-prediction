import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Task 1: Fix TotalCharges — convert text to number
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Task 2: Fill empty NaN values with median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Check it worked
print("Missing values:", df['TotalCharges'].isnull().sum())
print("Data type:", df['TotalCharges'].dtype)
# Save findings to text file
findings = """
=== WEEK 1 FINDINGS ===

1. Total customers: 7043
2. Churn rate: 26.5%
3. Month-to-month churn rate: 42%
4. Two-year contract churn rate: 3%
5. Average charge for churned customers: $74.44
6. Average charge for stayed customers: $61.27
7. Average tenure for churned customers: 17.98 months
8. Average tenure for stayed customers: 37.57 months
9. TotalCharges missing values fixed: 11
10. Zero missing values after cleaning
"""

with open('findings.txt', 'w') as f:
    f.write(findings)

print("findings.txt saved!")