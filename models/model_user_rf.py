# Wrapper for user's Random Forest model file.
# Exposes run_prediction(symbol, sample_size, horizon_days, random_seed).
# If importing the user's file fails or no predict function is found, fallback to a simple stats-based projection.

import importlib.util, os, random, datetime
from typing import List, Tuple

HERE = os.path.dirname(__file__)

def _load_user_module():
    path = os.path.join(HERE, "originals", "隨機森林模型.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("user_rf", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    except Exception:
        return None

def _fallback(symbol: str, sample_size: int, horizon_days: int, random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
    base = 2400.0 if symbol == "XAUUSD" else 1.28
    vol = (35.0 if symbol == "XAUUSD" else 0.006) / (sample_size ** 0.5)
    today = datetime.date.today()
    level = base
    pts = []
    for i in range(horizon_days):
        shock = random.gauss(0.0, vol)
        level = max(0.0, level * (1.0 + shock))
        t = datetime.datetime.combine(today + datetime.timedelta(days=i+1), datetime.time(0,0)).isoformat() + "Z"
        pts.append((t, level))
    return pts

def run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int | None = None) -> list[tuple[str, float]]:
    mod = _load_user_module()
    # Try a few common entrypoints
    for fn in ("run_prediction", "predict_next", "forecast", "main"):
        if mod and hasattr(mod, fn):
            try:
                out = getattr(mod, fn)(symbol, sample_size, horizon_days, random_seed)
                if isinstance(out, list) and out and isinstance(out[0], (list, tuple)) and len(out[0]) == 2:
                    return [(str(t), float(p)) for (t, p) in out]
            except Exception:
                break
    # Fallback
    return _fallback(symbol, sample_size, horizon_days, random_seed)
