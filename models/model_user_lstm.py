import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class DummyLSTM:
    def predict(self, X):
        last = X[:,-1,0]
        return last + np.random.normal(0,1, size=last.shape)

model = DummyLSTM()

def run_prediction(symbol, sample_size, horizon_days=7, random_seed=None):
    np.random.seed(random_seed or 42)

    last_price = 1700 if symbol=="XAUUSD" else 1.25
    X = np.array([last_price + np.random.normal(0,1,sample_size)]).reshape(1,sample_size,1)
    print("=== Debug LSTM Input X ===", X[:,:5,:])

    y_pred_scaled = model.predict(X)
    scaler = MinMaxScaler(feature_range=(last_price*0.9,last_price*1.1))
    scaler.fit(X.reshape(-1,1))
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_pred = np.clip(y_pred,0,None)

    start_time = datetime.utcnow()
    pts = []
    for i in range(horizon_days):
        t = (start_time + timedelta(days=i)).isoformat()+"Z"
        pts.append((t, y_pred[i % len(y_pred)]))

    print("=== Debug LSTM pts ===", pts[:10])
    return pts
