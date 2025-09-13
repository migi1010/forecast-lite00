from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from models.my_model import MyModel

# -------------------------------
# 初始化 FastAPI
# -------------------------------
app = FastAPI(title="Forecast Lite API")

# -------------------------------
# 載入模型
# -------------------------------
# 兩個 h5 模型
model1 = tf.keras.models.load_model("models/inception_model_final.h5")
model2 = tf.keras.models.load_model("models/lstm_model.h5")

# 自訂 py 模型
model3 = MyModel()
# 如果需要載入權重，例如 h5:
# model3.load_weights("models/model3_weights.h5")

# -------------------------------
# 輸入資料格式
# -------------------------------
class InputData(BaseModel):
    data: list

# -------------------------------
# API 端點
# -------------------------------
@app.post("/predict/model1")
def predict_model1(input_data: InputData):
    x = np.array(input_data.data)
    pred = model1.predict(x)
    return {"prediction": pred.tolist()}

@app.post("/predict/model2")
def predict_model2(input_data: InputData):
    x = np.array(input_data.data)
    pred = model2.predict(x)
    return {"prediction": pred.tolist()}

@app.post("/predict/model3")
def predict_model3(input_data: InputData):
    x = np.array(input_data.data)
    pred = model3.predict(x)
    return {"prediction": pred.tolist()}

# -------------------------------
# 測試用首頁
# -------------------------------
@app.get("/")
def root():
    return {"message": "Forecast Lite API is running!"}
