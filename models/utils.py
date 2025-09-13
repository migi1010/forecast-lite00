import yfinance as yf

def fetch_xauusd_history(period="60d", interval="1h"):
    """
    取得 XAUUSD 黃金歷史收盤價
    period: 最近多少天
    interval: '1h' 每小時, '1d' 每日
    """
    symbol = "GC=F"  # Yahoo Finance 黃金期貨
    df = yf.download(symbol, period=period, interval=interval)
    df = df[['Close']].dropna()
    df.index = df.index.tz_localize(None)  # 移除時區
    return df
