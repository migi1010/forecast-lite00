import numpy as np
from datetime import datetime, timedelta
from models.utils import fetch_xauusd_history


# Dummy 隨機森林模型
class DummyRF:
    def predict(self, X):
        return X[:,-1,0] + np.random.normal(0, 1.5, size=X.shape[0])

model = DummyRF()

def run_prediction(symbol, sample_size, horizon_days=7, random_seed=None):
    np.random.seed(random_seed or 42)
    last_prices = fetch_xauusd_history()['Close'].values[-sample_size:] if symbol=="XAUUSD" else np.ones(sample_size)*1.25
    X = last_prices.reshape(1, sample_size, 1)
    y_pred = model.predict(X)
    
    start_time = datetime.utcnow()
    pts = []
    for i in range(horizon_days):
        t = (start_time + timedelta(days=i)).isoformat()+"Z"
        pts.append((t, float(y_pred[i % len(y_pred)])))
    return pts
