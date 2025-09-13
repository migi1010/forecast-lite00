import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from .utils import fetch_xauusd_history

# 載入你的 LSTM h5 模型
model = load_model("models/lstm_model.h5")

def run_prediction(symbol, sample_size, horizon_days=7, random_seed=None):
    np.random.seed(random_seed or 42)
    df = fetch_xauusd_history() if symbol=="XAUUSD" else None
    data = df['Close'].values[-sample_size:] if df is not None else np.ones(sample_size)*1.25

    # 標準化
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data.reshape(-1,1)).reshape(1, sample_size, 1)

    # LSTM 預測
    pred_scaled = model.predict(data_scaled).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

    start_time = datetime.utcnow()
    pts = [( (start_time + timedelta(days=i)).isoformat()+"Z", float(pred[i % len(pred)]) )
           for i in range(horizon_days)]
    return pts
