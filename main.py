import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# -----------------------------
# 載入模型
# -----------------------------
# Random Forest 模型（自訂的 DummyRF 或你原本的 h5 RF 模型）
from models.model_user_rf import run_prediction as run_rf

# LSTM 與 Inception 模型
lstm_model = load_model("models/lstm_model.h5")
inception_model = load_model("models/inception_model_final.h5")

# -----------------------------
# 抓即時 XAUUSD 資料
# -----------------------------
def fetch_xauusd_history(period="60d", interval="1h"):
    ticker = yf.Ticker("XAUUSD=X")
    df = ticker.history(period=period, interval=interval)
    df = df[["Open", "High", "Low", "Close", "Volume"]]  # 五個特徵
    return df

# -----------------------------
# 預測函數
# -----------------------------
def predict_all(sample_size=30, horizon_days=7):
    df = fetch_xauusd_history()
    last_sequence = df.values[-sample_size:]  # 取最後 sample_size 筆資料
    
    # -------------------------
    # Random Forest 預測 (假設 RF 只需要 Close)
    # -------------------------
    rf_pts = run_rf("XAUUSD", sample_size, horizon_days=horizon_days)
    
    # -------------------------
    # LSTM 預測
    # -------------------------
    X_lstm = last_sequence.reshape(1, sample_size, 5)
    lstm_pred = lstm_model.predict(X_lstm)
    
    # -------------------------
    # Inception 預測
    # -------------------------
    X_incep = last_sequence.reshape(1, sample_size, 5)
    inception_pred = inception_model.predict(X_incep)
    
    # -------------------------
    # 回傳結果
    # -------------------------
    start_time = datetime.utcnow()
    results = []
    for i in range(horizon_days):
        t = (start_time + timedelta(days=i)).isoformat() + "Z"
        results.append({
            "time": t,
            "RF": float(rf_pts[i % len(rf_pts)]),
            "LSTM": float(lstm_pred[0][i % lstm_pred.shape[1]]),
            "Inception": float(inception_pred[0][i % inception_pred.shape[1]])
        })
    
    return results

# -----------------------------
# 測試執行
# -----------------------------
if __name__ == "__main__":
    preds = predict_all()
    for p in preds:
        print(p)
