import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Load model and the saved training columns
model = joblib.load('fraud_model.pkl')
model_columns = joblib.load('model_columns.pkl')


@app.post('/predict')
def predict_fraud(transaction: dict):
    # 1. Convert incoming JSON dictionary to a DataFrame
    df_input = pd.DataFrame([transaction])

    # 2. Apply the same one-hot encoding as your training pipeline
    df_input = pd.get_dummies(df_input)

    # 3. Reindex to match the EXACT columns the model was trained on
    # (This automatically adds missing columns as 0 and drops extra ones)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # 4. Get the probability of fraud
    fraud_prob = float(model.predict_proba(df_input)[:, 1][0])
    is_fraud = fraud_prob >= 0.8

    return {
        'fraud_probability': round(fraud_prob, 4),
        'flagged_as_fraud': bool(is_fraud),
    }