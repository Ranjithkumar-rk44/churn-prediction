import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('charts', exist_ok=True)

# Load the data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── Q1: How many customers churned? ──────────────────
print("=== CHURN COUNT ===")
print(df['Churn'].value_counts())

print("\n=== CHURN PERCENTAGE ===")
print(df['Churn'].value_counts(normalize=True) * 100)

# ── Q2: Any missing values? ───────────────────────────
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ── Q3: Basic statistics ──────────────────────────────
print("\n=== BASIC STATS ===")
print(df.describe())

# ── Q4: Churn by Contract Type ───────────────────────
print("\n=== CHURN BY CONTRACT ===")
print(df.groupby('Contract')['Churn'].value_counts())

# Chart 1: Churn Count
plt.figure(figsize=(6,4))
df['Churn'].value_counts().plot(kind='bar', color=['green','red'])
plt.title('Total Churn Count')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('charts/churn_count.png')
plt.show()
print("Chart 1 saved!")

# Chart 2: Churn by Contract Type
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.savefig('charts/churn_by_contract.png')
plt.show()
print("Chart 2 saved!")

# Chart 3: Tenure vs Churn
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title('Churn by Tenure (months)')
plt.tight_layout()
plt.savefig('charts/churn_by_tenure.png')
plt.show()
print("Chart 3 saved!")

print("\n=== ALL CHARTS SAVED TO charts/ FOLDER ===")
