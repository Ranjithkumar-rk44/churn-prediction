import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Success! Rows and columns:", df.shape)
print(df.head())