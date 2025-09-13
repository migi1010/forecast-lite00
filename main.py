from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import load_model
from models.model_user_rf import run_prediction as run_user_rf

# ---------------------------
# 初始化 FastAPI
# ---------------------------
app = FastAPI(title="Forecast Lite API")

# ---------------------------
# 載入 H5 模型
# ---------------------------
lstm_model = load_model("models/lstm_model.h5")
inception_model = load_model("models/inception_model_final.h5")

# ---------------------------
# 請求模型
# ---------------------------
class PredictRequest(BaseModel):
    symbol: str
    sample_size: int
    horizon_days: int = 7

class PredictionPoint(BaseModel):
    timestamp: str
    lstm: float
    inception: float
    user_rf: float

# ---------------------------
# Helper 函數
# ---------------------------
def run_keras_model(model, last_prices):
    X = np.array(last_prices).reshape(1, len(last_prices), 1)
    y_pred = model.predict(X)
    return y_pred.flatten()

# ---------------------------
# API 路由
# ---------------------------
@app.post("/predict", response_model=List[PredictionPoint])
def predict(req: PredictRequest):
    # 假資料或從資料庫抓取歷史價格
    last_prices = np.random.random(req.sample_size) * 2000  # 你可以改成真實 fetch_xauusd_history
    
    # 三個模型的預測
    lstm_pred = run_keras_model(lstm_model, last_prices)
    inception_pred = run_keras_model(inception_model, last_prices)
    user_rf_pred = np.array([p for t, p in run_user_rf(req.symbol, req.sample_size, req.horizon_days)])
    
    # 組成時間序列
    start_time = datetime.utcnow()
    results = []
    for i in range(req.horizon_days):
        t = (start_time + timedelta(days=i)).isoformat() + "Z"
        results.append(PredictionPoint(
            timestamp=t,
            lstm=float(lstm_pred[i % len(lstm_pred)]),
            inception=float(inception_pred[i % len(inception_pred)]),
            user_rf=float(user_rf_pred[i % len(user_rf_pred)])
        ))
    return results
