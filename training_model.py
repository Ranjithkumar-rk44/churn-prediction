import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop(columns=['customerID'])

# Step 2: Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Step 3: Fix Churn to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step 4: Encode ALL text columns to numbers at once
df = pd.get_dummies(df, drop_first=True)

print("Columns after encoding:", df.shape)

# Step 5: Separate X and y
X = df.drop(columns=['Churn'])
y = df['Churn']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Step 6: Split 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 7: Scale numbers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Test model
predictions = model.predict(X_test)
print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(classification_report(y_test, predictions))