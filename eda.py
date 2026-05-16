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
# ═══════════════════════════════════════
# DAY 4 - DEEPER ANALYSIS & HEATMAP
# ═══════════════════════════════════════

# Chart 4: Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.tight_layout()
plt.savefig('charts/monthly_charges_churn.png')
plt.show()
print("Chart 4 saved!")

# Chart 5: Tenure vs Churn boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')
plt.tight_layout()
plt.savefig('charts/tenure_churn.png')
plt.show()
print("Chart 5 saved!")

# Chart 6: Churn by Internet Service
plt.figure(figsize=(8,5))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service Type')
plt.tight_layout()
plt.savefig('charts/churn_by_internet.png')
plt.show()
print("Chart 6 saved!")

# Chart 7: Correlation Heatmap
df2 = df.copy()
df2['Churn'] = df2['Churn'].map({'Yes': 1, 'No': 0})
df2['TotalCharges'] = pd.to_numeric(df2['TotalCharges'], errors='coerce')

num_cols = df2[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]

plt.figure(figsize=(8,5))
sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('charts/correlation_heatmap.png')
plt.show()
print("Chart 7 saved!")

# Print key findings
print("\n=== DAY 4 KEY FINDINGS ===")
print("Average monthly charge for churned customers:",
      round(df[df['Churn']=='Yes']['MonthlyCharges'].mean(), 2))
print("Average monthly charge for stayed customers:",
      round(df[df['Churn']=='No']['MonthlyCharges'].mean(), 2))
print("Average tenure for churned customers:",
      round(df[df['Churn']=='Yes']['tenure'].mean(), 2))
print("Average tenure for stayed customers:",
      round(df[df['Churn']=='No']['tenure'].mean(), 2))
