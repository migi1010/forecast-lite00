from fastapi import FastAPI
import uvicorn
import tensorflow as tf
import numpy as np
from models.utils import fetch_xauusd_history

app = FastAPI()

# 載入三個模型
rf_model = None  # 如果你的 RF 是用 pickle 之類的，可以這裡載入
lstm_model = tf.keras.models.load_model("lstm_model.h5")
inception_model = tf.keras.models.load_model("inception_model_final.h5")


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.get("/predict/lstm")
def predict_lstm(symbol: str = "XAUUSD", sample_size: int = 60, horizon_days: int = 7):
    history = fetch_xauusd_history()
    last_prices = history["Close"].values[-sample_size:]
    X = last_prices.reshape(1, sample_size, 1)
    y_pred = lstm_model.predict(X).flatten()
    return {"symbol": symbol, "predictions": y_pred[:horizon_days].tolist()}


@app.get("/predict/inception")
def predict_inception(symbol: str = "XAUUSD", sample_size: int = 60, horizon_days: int = 7):
    history = fetch_xauusd_history()
    last_prices = history["Close"].values[-sample_size:]
    X = last_prices.reshape(1, sample_size, 1)
    y_pred = inception_model.predict(X).flatten()
    return {"symbol": symbol, "predictions": y_pred[:horizon_days].tolist()}


# 這裡可以再加 RF 的 predict API
@app.get("/predict/rf")
def predict_rf(symbol: str = "XAUUSD", sample_size: int = 60, horizon_days: int = 7):
    return {"symbol": symbol, "predictions": "RF 還沒接上 🚧"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
