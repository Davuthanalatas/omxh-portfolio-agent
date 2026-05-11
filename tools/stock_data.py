import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

OMXH_STOCKS = [
    "NOKIA.HE", "NESTE.HE", "FORTUM.HE", "STERV.HE",
    "KNEBV.HE", "OUT1V.HE", "METSO.HE", "TIETO.HE",
    "ELISA.HE", "WRT1V.HE"
]

def fetch_stock_data(ticker: str, period: str = "6mo", end_date: str = None) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    effective_end = end_date or os.environ.get("BACKTEST_END_DATE")
    if effective_end:
        end_dt   = datetime.strptime(effective_end, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=180)
        df = stock.history(start=start_dt.strftime("%Y-%m-%d"), end=effective_end)
    else:
        df = stock.history(period=period)
    df.dropna(inplace=True)
    return df

def fetch_multiple_stocks(tickers: list = OMXH_STOCKS, period: str = "6mo", end_date: str = None) -> dict:
    effective_end = end_date or os.environ.get("BACKTEST_END_DATE")
    data = {}
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period, effective_end)
            if not df.empty:
                data[ticker] = df
                print(f"✓ {ticker} — {len(df)} rows")
            else:
                print(f"✗ {ticker} — no data")
        except Exception as e:
            print(f"✗ {ticker} — error: {e}")
    return data

if __name__ == "__main__":
    print("Fetching OMXH stock data...\n")
    stocks = fetch_multiple_stocks()
    print(f"\nSuccessfully loaded {len(stocks)} stocks.")
