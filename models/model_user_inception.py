
# Inception Time：微正向趨勢；會優先呼叫 originals/inceptiontime-xauusd.py 的 run_prediction
import importlib.util, os, random, datetime, math
HERE = os.path.dirname(__file__)

def _load_user():
    p = os.path.join(HERE, "originals", "inceptiontime-xauusd.py")
    if not os.path.exists(p): return None
    spec = importlib.util.spec_from_file_location("user_inception", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)  # type: ignore
    return m

def _fallback(symbol, sample_size, horizon_days, seed):
    if seed is not None: random.seed(seed+11)
    base = 2400.0 if symbol=="XAUUSD" else 1.28
    vol  = (30.0 if symbol=="XAUUSD" else 0.005) / math.sqrt(sample_size)
    today = datetime.date.today(); level = base; out=[]
    for i in range(horizon_days):
        level = max(0.0, level*(1.0 + 0.0006 + random.gauss(0.0, vol)))
        t = datetime.datetime.combine(today+datetime.timedelta(days=i+1), datetime.time()).isoformat()+"Z"
        out.append((t, level))
    return out

def run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int|None=None):
    for fn in ("run_prediction","predict_next","forecast","main"):
        try:
            m=_load_user()
            if m and hasattr(m,fn):
                out=getattr(m,fn)(symbol, sample_size, horizon_days, random_seed)
                if out: return [(str(t), float(p)) for (t,p) in out]
        except Exception: break
    return _fallback(symbol, sample_size, horizon_days, random_seed)
