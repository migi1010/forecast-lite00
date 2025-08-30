from fastapi import FastAPI, Query
import pandas as pd
import tensorflow as tf
import joblib
import requests

# ===== 1. 建立 FastAPI =====
app = FastAPI(title="Forecast Lite API with Live Data")

# ===== 2. 載入模型 =====
models = {
    "inception": tf.keras.models.load_model("models/inception_model.h5", compile=False),
    "lstm": tf.keras.models.load_model("models/lstm_model.h5", compile=False),
    "rf": joblib.load("models/rf_model.pkl")
}

# ===== 3. Alpha Vantage API Key =====
API_KEY = "FUGXBR5LTNZSFCVT"

# ===== 4. 取得即時價格函式 =====
def get_latest_rate(symbol: str):
    if symbol.upper() == "XAUUSD":
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={API_KEY}"
    elif symbol.upper() == "GBPUSD":
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=GBP&to_currency=USD&apikey={API_KEY}"
    else:
        return None

    resp = requests.get(url).json()
    try:
        rate = float(resp["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
    except:
        rate = None
    return rate

# ===== 5. 模擬歷史資料 (前 n_samples 天) =====
def get_recent_data(symbol: str, n_samples: int):
    latest = get_latest_rate(symbol)
    if latest is None:
        raise ValueError("無法取得即時價格")
    # 模擬特徵
    data = pd.DataFrame({
        "Close": [latest]*n_samples
    })
    data["Datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=n_samples)
    return data

# ===== 6. 前處理函式 =====
def preprocess(data: pd.DataFrame):
    X = data.drop(columns=["Datetime"]).values
    return X.reshape((1, X.shape[0], X.shape[1]))

# ===== 7. 單一模型預測 =====
@app.get("/predict")
def predict(
    symbol: str = Query("XAUUSD", description="資產: XAUUSD / GBPUSD"),
    model_name: str = Query("inception", description="模型名稱: inception / lstm / rf"),
    n_samples: int = Query(300, ge=1, description="樣本大小 (天數)")
):
    data = get_recent_data(symbol, n_samples)
    X = preprocess(data)

    model = models.get(model_name)
    if model is None:
        return {"error": f"模型 {model_name} 不存在，可選擇: {list(models.keys())}"}

    if model_name == "rf":
        X_rf = X.reshape(X.shape[1], X.shape[2])
        y_pred = model.predict([X_rf.flatten()])
    else:
        y_pred = model.predict(X)

    return {
        "symbol": symbol.upper(),
        "model": model_name,
        "n_samples": n_samples,
        "prediction": y_pred.tolist(),
        "latest_rate": data["Close"].iloc[-1]
    }

# ===== 8. 三模型同時預測 =====
@app.get("/predict_all")
def predict_all(symbol: str = Query("XAUUSD", description="資產: XAUUSD / GBPUSD"), n_samples: int = Query(300, ge=1)):
    data = get_recent_data(symbol, n_samples)
    X = preprocess(data)

    results = {}
    for name, model in models.items():
        if name == "rf":
            X_rf = X.reshape(X.shape[1], X.shape[2])
            pred = model.predict([X_rf.flatten()])
        else:
            pred = model.predict(X)
        results[name] = pred.tolist()

    return {
        "symbol": symbol.upper(),
        "n_samples": n_samples,
        "results": results,
        "latest_rate": data["Close"].iloc[-1]
    }

# ===== 9. 健康檢查 =====
@app.get("/")
def root():
    return {"message": "Forecast Lite API 運行中"}
