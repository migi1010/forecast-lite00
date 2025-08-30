import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from tensorflow.keras.models import load_model
import requests

app = FastAPI()

# Alpha Vantage API Key
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# 載入模型
lstm_model = load_model("models/lstm_model.h5")
inception_model = load_model("models/inception_model.h5")
rf_model = joblib.load("models/rf_model.pkl")

# 抓即時資料
def fetch_data(symbol: str, n_samples: int):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url).json()
    df = pd.DataFrame(r.get("Time Series (Daily)", {})).T.astype(float)
    df = df.sort_index(ascending=True)
    return df.iloc[-n_samples:]

# 單模型預測
@app.get("/predict")
def predict(symbol: str, model_name: str, n_samples: int = 300):
    data = fetch_data(symbol, n_samples)
    X = data.values
    if model_name.lower() == "lstm":
        pred = lstm_model.predict(np.expand_dims(X, axis=0))
    elif model_name.lower() == "inception":
        pred = inception_model.predict(np.expand_dims(X, axis=0))
    elif model_name.lower() == "rf":
        pred = rf_model.predict(X)
    else:
        return {"error": "Unknown model_name"}
    return {"symbol": symbol, "model": model_name, "prediction": pred.tolist()}

# 三模型一次預測
@app.get("/predict_all")
def predict_all(symbol: str, n_samples: int = 300):
    data = fetch_data(symbol, n_samples)
    X = data.values
    return {
        "symbol": symbol,
        "predictions": {
            "lstm": lstm_model.predict(np.expand_dims(X, axis=0)).tolist(),
            "inception": inception_model.predict(np.expand_dims(X, axis=0)).tolist(),
            "rf": rf_model.predict(X).tolist()
        }
    }
