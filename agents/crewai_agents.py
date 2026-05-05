import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
from tools.stock_data import fetch_multiple_stocks
from tools.analysis import analyze_stock
import pandas as pd

load_dotenv()

# ── TOOLS ──────────────────────────────────────────────────────────────────
# Tools are plain Python functions wrapped with @tool
# The agent calls these when it needs real-world data

@tool("Fetch OMXH Stock Data")
def fetch_omxh_data(tickers: str) -> str:
    """Fetches historical stock data for given OMXH tickers (comma-separated)."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    stocks = fetch_multiple_stocks(ticker_list)
    return f"Successfully fetched data for: {list(stocks.keys())}"

@tool("Analyze OMXH Stocks")
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

data_agent = Agent(
    role="Stock Data Collector",
    goal="Fetch accurate and up-to-date stock data for OMXH listed companies",
    backstory="""You are a financial data specialist with deep expertise in 
    Nordic stock markets. You know how to retrieve reliable market data and 
    ensure it is clean and ready for analysis.""",
    tools=[fetch_omxh_data],
    verbose=True
)

analysis_agent = Agent(
    role="Financial Analyst",
    goal="Analyze stock metrics and identify the best investment opportunities",
    backstory="""You are a senior financial analyst specializing in quantitative 
    analysis of Nordic equities. You interpret RSI, moving averages, Sharpe ratio, 
    and volatility to assess risk and return profiles of stocks.""",
    tools=[analyze_omxh_stocks],
    verbose=True
)

report_agent = Agent(
    role="Portfolio Strategist",
    goal="Generate a clear, actionable portfolio recommendation based on the analysis",
    backstory="""You are an experienced portfolio manager who translates complex 
    financial analysis into simple, actionable investment strategies. You communicate 
    clearly and always justify your recommendations with data.""",
    verbose=True
)

# ── TASKS ──────────────────────────────────────────────────────────────────
# Tasks define what each agent actually does
# Each task has a description, expected output, and is assigned to one agent

TICKERS = "NOKIA.HE, NESTE.HE, FORTUM.HE, STERV.HE, KNEBV.HE, OUT1V.HE, METSO.HE, TIETO.HE, ELISA.HE, WRT1V.HE"

task_fetch = Task(
    description=f"Fetch stock data for the following OMXH tickers: {TICKERS}",
    expected_output="Confirmation that data was fetched successfully for all tickers.",
    agent=data_agent
)

task_analyze = Task(
    description=f"""Analyze the following OMXH stocks: {TICKERS}
    Calculate RSI, MA20, MA50, Sharpe ratio, and volatility for each.
    Identify which stocks are overbought, oversold, or well-positioned.""",
    expected_output="A detailed table of metrics with interpretation for each stock.",
    agent=analysis_agent
)

task_report = Task(
    description="""Based on the analysis, generate a portfolio recommendation.
    Clearly state which stocks to BUY, HOLD, or AVOID.
    Justify each decision using the metrics from the analysis.
    Keep the report concise and professional.""",
    expected_output="A structured portfolio recommendation report with BUY/HOLD/AVOID ratings.",
    agent=report_agent
)

# ── CREW ───────────────────────────────────────────────────────────────────
# The Crew brings all agents and tasks together
# Process.sequential means agents work one after another (like a pipeline)

crew = Crew(
    agents=[data_agent, analysis_agent, report_agent],
    tasks=[task_fetch, task_analyze, task_report],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("Starting OMXH Portfolio Analysis with CrewAI...\n")
    result = crew.kickoff()
    print("\n===== FINAL REPORT =====\n")
    print(result)

    os.makedirs("output", exist_ok=True)
    with open("output/crewai_report.txt", "w") as f:
        f.write(str(result))
    print("\nReport saved to output/crewai_report.txt")