
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import importlib.util, os, glob, statistics, datetime

APP_DIR   = os.path.dirname(__file__)
MODELS_DIR= os.path.join(APP_DIR, "models")
STATIC_DIR= os.path.join(APP_DIR, "static")

app = FastAPI(title="Forecast Lite", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# 靜態檔存在才掛，避免沒有 static/ 就崩潰
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def root_index():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return JSONResponse({"ok": True, "msg": "Backend OK. Upload static/index.html to serve UI."})

class PredictIn(BaseModel):
    symbol: str = Field(pattern="^(XAUUSD|GBPUSD)$")
    model: str
    sample_size: int
    horizon_days: int = 7
    random_seed: int | None = None

    @field_validator("sample_size")
    def _ss(cls, v):
        if v not in (300, 3000, 30000): raise ValueError("sample_size must be one of 300, 3000, 30000")
        return v

def _discover_models():
    items = []
    for f in sorted(glob.glob(os.path.join(MODELS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if name == "__init__": continue
        spec = importlib.util.spec_from_file_location(name, f)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "run_prediction"):
                items.append(name)
        except Exception:
            continue
    return items

@app.get("/health")
def health(): return {"ok": True, "time": datetime.datetime.utcnow().isoformat()+"Z"}

@app.get("/models")
def list_models(): return {"models": _discover_models()}

@app.post("/predict")
def predict(payload: PredictIn):
    # 英鎊先不開放：外觀殼
    if payload.symbol == "GBPUSD":
        raise HTTPException(status_code=501, detail="GBPUSD 暫不支援（外觀預覽中）")

    available = _discover_models()
    if payload.model not in available:
        raise HTTPException(status_code=400, detail=f"Model '{payload.model}' not found. Available: {available}")

    mod_path = os.path.join(MODELS_DIR, f"{payload.model}.py")
    spec = importlib.util.spec_from_file_location(payload.model, mod_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    pts = mod.run_prediction(payload.symbol, payload.sample_size, payload.horizon_days, payload.random_seed)
    points = [{"t": t, "price": float(p)} for (t, p) in pts]
    prices = [p["price"] for p in points]
    return {
        "symbol": payload.symbol,
        "model": payload.model,
        "horizon_days": payload.horizon_days,
        "generated_at": datetime.datetime.utcnow().isoformat()+"Z",
        "points": points,
        "stats": {
            "min": min(prices) if prices else None,
            "max": max(prices) if prices else None,
            "start": prices[0] if prices else None,
            "end": prices[-1] if prices else None,
            "mean": statistics.fmean(prices) if prices else None,
        },
    }
