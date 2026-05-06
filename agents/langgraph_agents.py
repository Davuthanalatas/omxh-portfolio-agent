import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
from tools.stock_data import fetch_multiple_stocks
from tools.analysis import analyze_stock
import pandas as pd
import time

load_dotenv()
client = OpenAI()

TICKERS = [
    "NOKIA.HE", "NESTE.HE", "FORTUM.HE", "STERV.HE",
    "KNEBV.HE", "OUT1V.HE", "METSO.HE", "TIETO.HE",
    "ELISA.HE", "WRT1V.HE"
]

# ── STATE ──────────────────────────────────────────────────────────────────
# This is the shared whiteboard all nodes read from and write to.
# TypedDict enforces what fields are allowed in the state.

class PortfolioState(TypedDict):
    tickers: list[str]
    raw_data: dict[str, Any]
    analysis_table: str
    report: str
    start_time: float

# ── NODE 1: DATA NODE ──────────────────────────────────────────────────────
# Fetches raw stock data and stores it in state.
# Equivalent to the Data Agent in CrewAI.

def data_node(state: PortfolioState) -> PortfolioState:
    print("\n[Node 1] Fetching stock data...")
    stocks = fetch_multiple_stocks(state["tickers"])
    print(f"  ✓ Loaded {len(stocks)} stocks")
    return {"raw_data": stocks}

# ── NODE 2: ANALYSIS NODE ──────────────────────────────────────────────────
# Reads raw_data from state, runs analysis, stores result as a string table.
# Equivalent to the Financial Analyst in CrewAI.

def analysis_node(state: PortfolioState) -> PortfolioState:
    print("\n[Node 2] Analyzing stocks...")
    results = []
    for ticker, df in state["raw_data"].items():
        analysis = analyze_stock(ticker, df)
        results.append(analysis)
        print(f"  ✓ {ticker} analyzed")

    df = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    table = df.to_string(index=False)
    return {"analysis_table": table}

# ── NODE 3: REPORT NODE ────────────────────────────────────────────────────
# Reads analysis_table from state, calls GPT-4o directly, writes the report.
# Equivalent to the Portfolio Strategist in CrewAI.
# Key difference: in LangGraph we call the LLM manually — full control.

def report_node(state: PortfolioState) -> PortfolioState:
    print("\n[Node 3] Generating portfolio report with GPT-4o...")

    prompt = f"""You are a professional portfolio strategist specializing in Nordic equity markets.

Based on the following OMXH stock analysis data, generate a structured portfolio recommendation report.

For each stock, assign a rating of BUY, HOLD, or AVOID.
Justify each decision using the provided metrics (RSI, moving averages, Sharpe ratio, volatility).
End with a clear summary table.

Analysis Data:
{state["analysis_table"]}

Write a professional, concise report."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior portfolio strategist for Nordic equity markets."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content
    elapsed = round(time.time() - state["start_time"], 2)
    print(f"  ✓ Report generated in {elapsed}s")
    return {"report": report}

# ── BUILD THE GRAPH ────────────────────────────────────────────────────────
# Here we wire the nodes together into a directed graph.
# set_entry_point defines where execution starts.
# add_edge defines the flow: data → analysis → report → END

def build_graph():
    graph = StateGraph(PortfolioState)

    graph.add_node("data_node", data_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("report_node", report_node)

    graph.set_entry_point("data_node")
    graph.add_edge("data_node", "analysis_node")
    graph.add_edge("analysis_node", "report_node")
    graph.add_edge("report_node", END)

    return graph.compile()

# ── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting OMXH Portfolio Analysis with LangGraph...\n")

    app = build_graph()

    initial_state: PortfolioState = {
        "tickers": TICKERS,
        "raw_data": {},
        "analysis_table": "",
        "report": "",
        "start_time": time.time()
    }

    result = app.invoke(initial_state)

    print("\n===== FINAL REPORT =====\n")
    print(result["report"])

    os.makedirs("output", exist_ok=True)
    with open("output/langgraph_report.txt", "w") as f:
        f.write(result["report"])
    print("\nReport saved to output/langgraph_report.txt")