# models/model_inception.py
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from .utils import fetch_xauusd_history

model = load_model("models/inception_model_final.h5")

def run_prediction(symbol, sample_size, horizon_days=7):
    data = fetch_xauusd_history()['Close'].values[-sample_size:]
    X = data.reshape(1, sample_size, 1)
    y_pred = model.predict(X)[0]
    
    start_time = datetime.utcnow()
    pts = []
    for i in range(horizon_days):
        t = (start_time + timedelta(days=i)).isoformat()+"Z"
        pts.append((t, float(y_pred[i % len(y_pred)])))
    return pts
