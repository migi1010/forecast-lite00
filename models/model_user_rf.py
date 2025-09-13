import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from .utils import fetch_xauusd_history

class DummyRF:
    def predict(self, X):
        return X[:,-1,0] + np.random.normal(0, 1.5, size=X.shape[0])

model = DummyRF()

def run_prediction(symbol, sample_size, horizon_days=7, random_seed=None):
    np.random.seed(random_seed or 42)
    last_prices = fetch_xauusd_history()['Close'].values[-sample_size:] if symbol=="XAUUSD" else np.ones(sample_size)*1.25
    X = last_prices.reshape(1, sample_size, 1)
    y_pred_scaled = model.predict(X)
    scaler = MinMaxScaler(feature_range=(X.min()*0.95, X.max()*1.05))
    scaler.fit(X.reshape(-1,1))
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_pred = np.clip(y_pred,0,None)
    start_time = datetime.utcnow()
    return [(start_time + timedelta(days=i)).isoformat()+"Z", y_pred[i % len(y_pred)]] for i in range(horizon_days)
