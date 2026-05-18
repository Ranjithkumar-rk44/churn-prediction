import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges first
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Feature 1: charge per tenure month
df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# Feature 2: is this customer paying above average?
avg_charge = df['MonthlyCharges'].mean()
df['above_avg_charge'] = (df['MonthlyCharges'] > avg_charge).astype(int)

# Feature 3: is this a long term customer?
df['is_long_term'] = (df['tenure'] > 24).astype(int)

print(df[['charge_per_tenure',
          'above_avg_charge',
          'is_long_term']].head(10))
print("\nNew columns created successfully!")