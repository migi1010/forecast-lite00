from fastapi import FastAPI
from models.model_user_rf import run_prediction as run_rf
from models.model_user_lstm import run_prediction as run_lstm
from models.model_user_inception import run_prediction as run_incep

app = FastAPI()

@app.get("/predict/rf")
def predict_rf():
    return run_rf("XAUUSD", sample_size=50, horizon_days=7)

@app.get("/predict/lstm")
def predict_lstm():
    return run_lstm("XAUUSD", sample_size=50, horizon_days=7)

@app.get("/predict/inception")
def predict_inception():
    return run_incep("XAUUSD", sample_size=50, horizon_days=7)
