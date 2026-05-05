import pandas as pd
import numpy as np

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calculate_moving_averages(df: pd.DataFrame) -> dict:
    return {
        "MA20": round(df["Close"].rolling(20).mean().iloc[-1], 4),
        "MA50": round(df["Close"].rolling(50).mean().iloc[-1], 4),
        "current_price": round(df["Close"].iloc[-1], 4)
    }

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
    daily_returns = df["Close"].pct_change().dropna()
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return round(sharpe, 4)

def calculate_volatility(df: pd.DataFrame) -> float:
    daily_returns = df["Close"].pct_change().dropna()
    return round(daily_returns.std() * np.sqrt(252) * 100, 2)

def analyze_stock(ticker: str, df: pd.DataFrame) -> dict:
    mas = calculate_moving_averages(df)
    return {
        "ticker": ticker,
        "current_price": mas["current_price"],
        "MA20": mas["MA20"],
        "MA50": mas["MA50"],
        "RSI": calculate_rsi(df),
        "sharpe_ratio": calculate_sharpe_ratio(df),
        "volatility_pct": calculate_volatility(df)
    }

if __name__ == "__main__":
    from stock_data import fetch_stock_data

    ticker = "NOKIA.HE"
    df = fetch_stock_data(ticker)
    result = analyze_stock(ticker, df)

    print(f"\nAnalysis for {ticker}:")
    for key, value in result.items():
        print(f"  {key}: {value}")
