import os
from fastapi import FastAPI
from tensorflow.keras.models import load_model

# 自訂模型 import
from models.model_user_rf import run_prediction as run_user_rf

app = FastAPI()

# 取得 port，Render 會提供環境變數 PORT
PORT = int(os.environ.get("PORT", 8000))

# 載入 h5 模型
lstm_model = load_model("models/lstm_model.h5")
inception_model = load_model("models/inception_model_final.h5")

@app.get("/")
def read_root():
    return {"message": "Forecast Lite0 is running!"}

@app.get("/predict/lstm")
def predict_lstm():
    # 假範例：輸出隨機結果
    import numpy as np
    return {"prediction": float(np.random.rand())}

@app.get("/predict/inception")
def predict_inception():
    import numpy as np
    return {"prediction": float(np.random.rand())}

@app.get("/predict/user_rf")
def predict_user_rf():
    result = run_user_rf(symbol="XAUUSD", sample_size=50)
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
