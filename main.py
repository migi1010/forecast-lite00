from fastapi import FastAPI
from models.model_user_rf import run_prediction as run_user_rf
from tensorflow.keras.models import load_model
import numpy as np

# 載入 H5 模型
model1 = load_model("models/inception_model_final.h5")
model2 = load_model("models/lstm_model.h5")

app = FastAPI()

@app.get("/predict")
def predict(symbol: str = "XAUUSD", sample_size: int = 10, horizon_days: int = 7):
    # 自訂 Python 模型預測
    pred_user_rf = run_user_rf(symbol, sample_size, horizon_days)

    # 範例 H5 模型預測
    last_prices = np.random.rand(sample_size)  # 這邊換成真實資料
    X = last_prices.reshape(1, sample_size, 1)
    pred_model1 = model1.predict(X).flatten().tolist()
    pred_model2 = model2.predict(X).flatten().tolist()

    return {
        "user_rf": pred_user_rf,
        "model1": pred_model1,
        "model2": pred_model2
    }
