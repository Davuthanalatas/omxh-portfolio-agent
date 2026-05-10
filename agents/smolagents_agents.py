import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from smolagents import CodeAgent, ToolCallingAgent, Tool, OpenAIServerModel
from dotenv import load_dotenv
from tools.stock_data import fetch_multiple_stocks
from tools.analysis import analyze_stock
import pandas as pd

load_dotenv()

TICKERS = [
    "NOKIA.HE", "NESTE.HE", "FORTUM.HE", "STERV.HE",
    "KNEBV.HE", "OUT1V.HE", "METSO.HE", "TIETO.HE",
    "ELISA.HE", "WRT1V.HE"
]

TICKERS_STR = ", ".join(TICKERS)

model = OpenAIServerModel(
    model_id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

class FetchStockDataTool(Tool):
    name = "fetch_stock_data"
    description = "Fetches historical stock data for OMXH tickers from Yahoo Finance."
    inputs = {
        "tickers": {
            "type": "string",
            "description": "Comma-separated list of OMXH ticker symbols"
        }
    }
    output_type = "string"

    def forward(self, tickers: str) -> str:
        ticker_list = [t.strip() for t in tickers.split(",")]
        stocks = fetch_multiple_stocks(ticker_list)
        return f"Successfully fetched data for: {list(stocks.keys())}"


class AnalyzeStocksTool(Tool):
    name = "analyze_stocks"
    description = "Analyzes OMXH stocks and returns RSI, moving averages, Sharpe ratio, and volatility."
    inputs = {
        "tickers": {
            "type": "string",
            "description": "Comma-separated list of OMXH ticker symbols to analyze"
        }
    }
    output_type = "string"

    def forward(self, tickers: str) -> str:
        ticker_list = [t.strip() for t in tickers.split(",")]
        stocks = fetch_multiple_stocks(ticker_list)
        results = []
        for ticker, df in stocks.items():
            analysis = analyze_stock(ticker, df)
            results.append(analysis)
        df = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
        return df.to_string(index=False)


# Sub-agents — passed directly to orchestrator without ManagedAgent wrapper
data_agent = ToolCallingAgent(
    name="stock_data_collector",
    tools=[FetchStockDataTool()],
    model=model,
    description="Fetches OMXH stock market data from Yahoo Finance."
)

analysis_agent = ToolCallingAgent(
    name="stock_analyzer",
    tools=[AnalyzeStocksTool()],
    model=model,
    description="Analyzes OMXH stocks and computes RSI, Sharpe ratio, moving averages, and volatility."
)

# Orchestrator receives sub-agents directly in managed_agents
orchestrator = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[data_agent, analysis_agent],
    description="Orchestrates stock data collection, analysis, and portfolio recommendation."
)

if __name__ == "__main__":
    print("Starting OMXH Portfolio Analysis with Smolagents...\n")
    start = time.time()

    task = f"""
    You are a portfolio strategist for Nordic equity markets.
    Complete these steps in order:
    1. Use stock_data_collector to fetch data for: {TICKERS_STR}
    2. Use stock_analyzer to analyze the same tickers and get their metrics
    3. Generate a professional portfolio recommendation report that:
       - Assigns BUY, HOLD, or AVOID to each of the 10 stocks
       - Justifies each decision using RSI, Sharpe ratio, moving averages, volatility
       - Ends with a clear summary table of all 10 stocks and their ratings
    """

    result = orchestrator.run(task)
    elapsed = round(time.time() - start, 2)

    print(f"\nCompleted in {elapsed}s")
    print("\n===== FINAL REPORT =====\n")
    print(result)

    os.makedirs("output", exist_ok=True)
    with open("output/smolagents_report.txt", "w") as f:
        f.write(str(result))
    print(f"\nReport saved to output/smolagents_report.txt")