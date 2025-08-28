def run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int | None = None):
    import random, datetime
    if random_seed is not None:
        random.seed(random_seed)
    base = 2400.0 if symbol == "XAUUSD" else 1.28
    today = datetime.date.today()
    pts = []
    level = base
    vol = (40.0 if symbol == "XAUUSD" else 0.008) / (sample_size ** 0.5)
    drift = -0.0005
    for i in range(horizon_days):
        shock = random.gauss(0.0, vol)
        level = max(0.0, level * (1.0 + drift + shock))
        t = datetime.datetime.combine(today + datetime.timedelta(days=i+1), datetime.time(0,0)).isoformat() + "Z"
        pts.append((t, level))
    return pts
