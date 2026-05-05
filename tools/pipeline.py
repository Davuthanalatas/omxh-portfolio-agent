import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stock_data import fetch_multiple_stocks
from tools.analysis import analyze_stock

def run_pipeline() -> pd.DataFrame:
    print("Fetching stock data...\n")
    stocks = fetch_multiple_stocks()

    print("\nAnalyzing stocks...\n")
    results = []
    for ticker, df in stocks.items():
        try:
            analysis = analyze_stock(ticker, df)
            results.append(analysis)
            print(f"✓ {ticker} analyzed")
        except Exception as e:
            print(f"✗ {ticker} — error: {e}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("sharpe_ratio", ascending=False)
    df_results = df_results.reset_index(drop=True)
    return df_results

if __name__ == "__main__":
    df = run_pipeline()
    print("\n--- OMXH Stock Analysis Summary ---\n")
    print(df.to_string(index=False))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/analysis_summary.csv", index=False)
    print("\nSaved to data/analysis_summary.csv")