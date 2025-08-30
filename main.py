from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import os
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# 讀取模型
inception_model = load_model("models/inception_model.h5")
lstm_model = load_model("models/lstm_model.h5")
rf_model = joblib.load("models/rf_model.pkl")

# Alpha Vantage API Key 從環境變數讀
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# 抓即時資料
def get_realtime_price(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&outputsize=compact"
    res = requests.get(url).json()
    ts = res.get("Time Series (Daily)", {})
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.sort_index()
    df = df.astype(float)
    return df

# 簡單特徵處理
def prepare_features(df, n_samples):
    df = df[-n_samples:]
    X = df['5. adjusted close'].values.reshape(-1,1)
    return X

# 單模型預測
@app.get("/predict")
def predict(symbol: str, model_name: str, n_samples: int = 300):
    df = get_realtime_price(symbol)
    X = prepare_features(df, n_samples)
    if model_name == "inception":
        pred = inception_model.predict(X[np.newaxis,:,:])
        pred_value = float(pred[0][0])
    elif model_name == "lstm":
        pred = lstm_model.predict(X[np.newaxis,:,:])
        pred_value = float(pred[0][0])
    elif model_name == "rf":
        pred_value = float(rf_model.predict(X[-1].reshape(1,-1))[0])
    else:
        return {"error": "invalid model_name"}
    latest_price = float(df['5. adjusted close'].iloc[-1])
    return {"symbol": symbol, "latest_rate": latest_price, "prediction": pred_value, "model_name": model_name, "n_samples": n_samples}

# 三模型預測
@app.get("/predict_all")
def predict_all(symbol: str, n_samples: int = 300):
    df = get_realtime_price(symbol)
    X = prepare_features(df, n_samples)
    inception_pred = float(inception_model.predict(X[np.newaxis,:,:])[0][0])
    lstm_pred = float(lstm_model.predict(X[np.newaxis,:,:])[0][0])
    rf_pred = float(rf_model.predict(X[-1].reshape(1,-1))[0])
    latest_price = float(df['5. adjusted close'].iloc[-1])
    return {
        "symbol": symbol,
        "latest_rate": latest_price,
        "n_samples": n_samples,
        "inception": inception_pred,
        "lstm": lstm_pred,
        "rf": rf_pred
    }
