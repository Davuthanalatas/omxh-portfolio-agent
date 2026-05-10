import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
from agents import Agent, Runner, function_tool
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

# ── TOOLS ──────────────────────────────────────────────────────────────────
# In the OpenAI SDK, tools are defined with @function_tool decorator.
# Much simpler than CrewAI's @tool or LangGraph's manual OpenAI calls.
# The SDK automatically converts the function signature into a tool schema.

@function_tool
def fetch_omxh_data(tickers: str) -> str:
    """Fetches historical stock data for given OMXH tickers (comma-separated)."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    stocks = fetch_multiple_stocks(ticker_list)
    return f"Successfully fetched data for: {list(stocks.keys())}"

@function_tool
def analyze_omxh_stocks(tickers: str) -> str:
    """Analyzes OMXH stocks and returns RSI, moving averages, Sharpe ratio, and volatility."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    stocks = fetch_multiple_stocks(ticker_list)
    results = []
    for ticker, df in stocks.items():
        analysis = analyze_stock(ticker, df)
        results.append(analysis)
    df = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    return df.to_string(index=False)

# ── AGENTS ─────────────────────────────────────────────────────────────────
# Notice how simple agent definition is in the SDK compared to CrewAI.
# No role, goal, backstory — just a name, instructions, and tools.
# The instructions ARE the agent's persona and objective combined.

TICKERS_STR = ", ".join(TICKERS)

data_agent = Agent(
    name="Stock_Data_Collector",
    instructions=f"""You are a financial data specialist for Nordic stock markets.
    Your job is to fetch stock data for these OMXH tickers: {TICKERS_STR}
    Use the fetch_omxh_data tool to retrieve the data.
    Once done, hand off to the Financial Analyst agent.""",
    model="gpt-4o",
    tools=[fetch_omxh_data]
)

analysis_agent = Agent(
    name="Financial_Analyst",
    instructions=f"""You are a senior financial analyst specializing in Nordic equities.
    Your job is to analyze these OMXH stocks: {TICKERS_STR}
    Use the analyze_omxh_stocks tool to get the metrics.
    Interpret RSI, moving averages, Sharpe ratio, and volatility for each stock.
    Identify which stocks are overbought, oversold, or well-positioned.
    Once done, hand off to the Portfolio Strategist agent.""",
    model="gpt-4o",
    tools=[analyze_omxh_stocks]
)

report_agent = Agent(
    name="Portfolio_Strategist",
    instructions="""You are an experienced portfolio manager for Nordic equity markets.
    Based on the analysis you received, generate a structured portfolio recommendation report.
    Clearly assign BUY, HOLD, or AVOID to each stock.
    Justify every decision using the specific metrics provided.
    End with a clear summary table.
    Keep the report professional and concise.""",
    model="gpt-4o",
    tools=[]
)

# ── HANDOFFS ───────────────────────────────────────────────────────────────
# This is what makes the SDK unique — agents hand off to each other.
# We define the handoff chain by adding handoffs to each agent.
# data_agent → analysis_agent → report_agent

data_agent.handoffs = [analysis_agent]
analysis_agent.handoffs = [report_agent]

# ── MAIN ───────────────────────────────────────────────────────────────────
# Runner.run() is async — it handles the full handoff chain automatically.
# We start with data_agent and it orchestrates the rest.

async def run_pipeline():
    print("Starting OMXH Portfolio Analysis with OpenAI Agents SDK...\n")
    start = time.time()

    result = await Runner.run(
        data_agent,
        input=f"Analyze these OMXH stocks and generate a portfolio recommendation: {TICKERS_STR}"
    )

    elapsed = round(time.time() - start, 2)
    print(f"\nCompleted in {elapsed}s")
    return result.final_output, elapsed

if __name__ == "__main__":
    report, elapsed = asyncio.run(run_pipeline())

    print("\n===== FINAL REPORT =====\n")
    print(report)

    os.makedirs("output", exist_ok=True)
    with open("output/openai_sdk_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to output/openai_sdk_report.txt")
    