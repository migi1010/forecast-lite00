# Drop your three Python model files here

Each model file must expose a callable named `run_prediction` with the signature:
    run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int | None) -> list[tuple[str, float]]

Return a list of (ISO8601 timestamp string, price float) pairs for the next `horizon_days` days.
The backend will pass through `symbol` ('XAUUSD' or 'GBPUSD'), `sample_size` (300, 3000, 30000), and an optional `random_seed`.

Example minimal implementation:

def run_prediction(symbol: str, sample_size: int, horizon_days: int, random_seed: int | None = None):
    import random, datetime
    if random_seed is not None:
        random.seed(random_seed)
    base = 2400.0 if symbol == "XAUUSD" else 1.28
    today = datetime.date.today()
    pts = []
    level = base
    vol = 50.0 / (sample_size ** 0.5) if symbol == "XAUUSD" else 0.01 / (sample_size ** 0.5)
    for i in range(horizon_days):
        level += random.gauss(0, vol)
        t = datetime.datetime.combine(today + datetime.timedelta(days=i+1), datetime.time(0,0)).isoformat() + "Z"
        pts.append((t, max(0.0, level)))
    return pts
