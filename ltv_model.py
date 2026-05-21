import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Create LTV column
# LTV = how much a customer has paid = TotalCharges
df['LTV'] = df['TotalCharges']

print("=== LTV STATISTICS ===")
print("Average LTV: $", round(df['LTV'].mean(), 2))
print("Highest LTV: $", round(df['LTV'].max(), 2))
print("Lowest LTV:  $", round(df['LTV'].min(), 2))

print("\n=== LTV BY CHURN STATUS ===")
print(df.groupby('Churn')['LTV'].mean().round(2))

# Features for LTV prediction
features = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
X = df[features]
y = df['LTV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LTV model
ltv_model = LinearRegression()
ltv_model.fit(X_train, y_train)

# Test model
predictions = ltv_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\n=== LTV MODEL RESULTS ===")
print(f"Mean Absolute Error: ${round(mae, 2)}")
print("This means predictions are off by $", round(mae, 2), "on average")

# Save model
joblib.dump(ltv_model, 'models/ltv_model.pkl')
print("\nLTV model saved!")

# Show top 5 highest value customers at risk
df['churn_flag'] = (df['Churn'] == 'Yes').astype(int)
high_value_churners = df[df['Churn'] == 'Yes'].nlargest(5, 'LTV')
print("\n=== TOP 5 HIGH VALUE CUSTOMERS AT RISK ===")
print(high_value_churners[['customerID', 'LTV',
                            'MonthlyCharges', 'tenure']].to_string())