from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI()

class Customer(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        # Create a row with 30 zeros — same as training data
        feature_names = scaler.feature_names_in_
        data = pd.DataFrame([np.zeros(len(feature_names))],
                           columns=feature_names)

        # Fill in the values we have
        if 'tenure' in data.columns:
            data['tenure'] = customer.tenure
        if 'MonthlyCharges' in data.columns:
            data['MonthlyCharges'] = customer.MonthlyCharges
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = customer.TotalCharges
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = customer.SeniorCitizen

        # Scale and predict
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]

        return {
            "will_churn": bool(prediction),
            "churn_probability": round(float(probability), 3),
            "risk_level": "HIGH" if probability > 0.6 else
                          "MEDIUM" if probability > 0.3 else "LOW"
        }
    except Exception as e:
        return {"error": str(e)}