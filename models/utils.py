# models/utils.py
import pandas as pd
import yfinance as yf

def fetch_xauusd_history(period="60d", interval="1h"):
    """
    取得 XAU/USD 歷史與即時價格
    period: 最近多少天資料
    interval: K 線間隔
    """
    try:
        data = yf.download("XAUUSD=X", period=period, interval=interval)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print("抓取 XAU/USD 失敗:", e)
        return pd.DataFrame({"Close":[1.25]})  # fallback
