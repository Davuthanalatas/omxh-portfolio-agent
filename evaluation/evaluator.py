import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, timedelta

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── CONSTANTS ─────────────────────────────────────────────────────────────

TICKERS = [
    "NOKIA.HE", "NESTE.HE", "FORTUM.HE", "STERV.HE",
    "KNEBV.HE", "OUT1V.HE", "METSO.HE", "TIETO.HE",
    "ELISA.HE", "WRT1V.HE"
]

_today            = date.today()
BACKTEST_END_DATE = (_today - timedelta(days=90)).strftime("%Y-%m-%d")
TODAY_STR         = _today.strftime("%Y-%m-%d")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

FALLBACK_PATHS = {
    "CrewAI":     os.path.join(OUTPUT_DIR, "crewai_report.txt"),
    "LangGraph":  os.path.join(OUTPUT_DIR, "langgraph_report.txt"),
    "OpenAI SDK": os.path.join(OUTPUT_DIR, "openai_sdk_report.txt"),
    "Smolagents": os.path.join(OUTPUT_DIR, "smolagents_report.txt"),
}

SMOLAGENTS_TIMEOUT = 120  # seconds

# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — SYSTEM METRICS
# ══════════════════════════════════════════════════════════════════════════

def count_tokens_estimate(text: str) -> int:
    return len(text) // 4

def check_completeness(report: str) -> bool:
    upper = report.upper()
    return all(k in upper for k in ["BUY", "HOLD", "AVOID"])

# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — STRUCTURAL METRICS
# ══════════════════════════════════════════════════════════════════════════

def structural_metrics(report: str, tickers: list) -> dict:
    upper = report.upper()

    ticker_bases    = [t.split(".")[0] for t in tickers]
    stock_coverage  = sum(1 for b in ticker_bases if b in upper)

    metric_terms    = ["RSI", "SHARPE", "VOLATILITY", "MA20", "MA50", "MOVING AVERAGE"]
    metric_mentions = sum(1 for term in metric_terms if term in upper)

    word_count  = len(report.split())
    buy_count   = len(re.findall(r"\bBUY\b",   upper))
    hold_count  = len(re.findall(r"\bHOLD\b",  upper))
    avoid_count = len(re.findall(r"\bAVOID\b", upper))

    return {
        "stock_coverage":  stock_coverage,
        "metric_mentions": metric_mentions,
        "word_count":      word_count,
        "buy_count":       buy_count,
        "hold_count":      hold_count,
        "avoid_count":     avoid_count,
    }

# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════

def fetch_actual_returns(tickers: list, since_date: str) -> dict:
    """Fetch price return from since_date to today for every ticker."""
    returns = {}
    print(f"\n  Fetching actual returns from {since_date} → {TODAY_STR} ...")
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(start=since_date, end=TODAY_STR)
            if len(df) >= 2:
                ret = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
                returns[ticker] = round(float(ret), 2)
                print(f"    {ticker:<15}  {returns[ticker]:+.2f}%")
            else:
                print(f"    {ticker:<15}  no data")
        except Exception as e:
            print(f"    {ticker:<15}  error: {e}")
    return returns


def parse_recommendations(report: str, tickers: list) -> dict:
    """
    Extract BUY / HOLD / AVOID for each ticker using four fall-through strategies
    that together cover all four report formats (CrewAI, LangGraph, OpenAI SDK, Smolagents).
    """
    ratings = {t: None for t in tickers}
    bases   = {t: t.split(".")[0] for t in tickers}       # NOKIA.HE → NOKIA

    # ── Strategy 1: Markdown table row  | TICKER | BUY | (any extra columns OK) ──
    for ticker in tickers:
        pat = (
            r"\|\s*(?:" + re.escape(ticker) + r"|" + re.escape(bases[ticker]) + r")[^|]*"
            r"\|\s*(BUY|HOLD|AVOID)\b"
        )
        m = re.search(pat, report, re.IGNORECASE)
        if m:
            ratings[ticker] = m.group(1).upper()

    # ── Strategy 2: "Recommendation: RATING" within a few lines of ticker ──
    unmatched = [t for t in tickers if ratings[t] is None]
    if unmatched:
        lines = report.split("\n")
        for i, line in enumerate(lines):
            lu = line.upper()
            for ticker in [t for t in unmatched if ratings[t] is None]:
                if ticker.upper() in lu or bases[ticker].upper() in lu:
                    ctx = "\n".join(lines[i : i + 6]).upper()
                    m   = re.search(
                        r"\bRECOMMENDATION\w*\s*[:\*\-–]+\s*(BUY|HOLD|AVOID)\b", ctx
                    )
                    if m:
                        ratings[ticker] = m.group(1).upper()

    # ── Strategy 3: Section headers  "- BUY:" / "1. HOLD" with tickers below ──
    unmatched = [t for t in tickers if ratings[t] is None]
    if unmatched:
        lines          = report.split("\n")
        current_section = None
        for line in lines:
            stripped = line.strip().upper()
            sec_m = re.match(r"^[-#*\d.\s]*\b(BUY|HOLD|AVOID)\b[\s:]*$", stripped)
            if sec_m:
                current_section = sec_m.group(1)
                continue
            if current_section:
                for ticker in [t for t in unmatched if ratings[t] is None]:
                    if ticker.upper() in stripped or bases[ticker].upper() in stripped:
                        ratings[ticker] = current_section

    # ── Strategy 4: Nearest BUY/HOLD/AVOID word to each ticker mention ──
    unmatched = [t for t in tickers if ratings[t] is None]
    if unmatched:
        upper = report.upper()
        for ticker in unmatched:
            pos = upper.find(ticker.upper())
            if pos == -1:
                pos = upper.find(bases[ticker].upper())
            if pos == -1:
                continue
            ws, we     = max(0, pos - 400), min(len(upper), pos + 400)
            window     = upper[ws:we]
            ticker_pos = pos - ws
            best, dist = None, float("inf")
            for rating in ["BUY", "HOLD", "AVOID"]:
                for m in re.finditer(rf"\b{rating}\b", window):
                    d = abs(m.start() - ticker_pos)
                    if d < dist:
                        dist, best = d, rating
            ratings[ticker] = best

    return ratings


def score_backtest(recommendations: dict, actual_returns: dict) -> dict:
    buy_rets   = [actual_returns[t] for t, r in recommendations.items()
                  if r == "BUY"   and t in actual_returns]
    avoid_rets = [actual_returns[t] for t, r in recommendations.items()
                  if r == "AVOID" and t in actual_returns]

    avg_buy   = round(sum(buy_rets)   / len(buy_rets),   2) if buy_rets   else None
    avg_avoid = round(sum(avoid_rets) / len(avoid_rets), 2) if avoid_rets else None
    score     = round(avg_buy - avg_avoid, 2) if (avg_buy is not None and avg_avoid is not None) else None

    return {
        "avg_buy_return":   avg_buy,
        "avg_avoid_return": avg_avoid,
        "backtest_score":   score,
        "buy_tickers":      [t for t, r in recommendations.items() if r == "BUY"],
        "avoid_tickers":    [t for t, r in recommendations.items() if r == "AVOID"],
        "hold_tickers":     [t for t, r in recommendations.items() if r == "HOLD"],
    }

# ══════════════════════════════════════════════════════════════════════════
# FRAMEWORK RUNNERS
# ══════════════════════════════════════════════════════════════════════════

def _load_fallback(name: str) -> str:
    with open(FALLBACK_PATHS[name], "r") as f:
        return f.read()


def _run_with_timeout(func, timeout_seconds: int) -> tuple:
    """Returns (result, elapsed, status).  status: 'ok' | 'timeout' | 'error:<msg>'"""
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func)
        try:
            result = future.result(timeout=timeout_seconds)
            return result, round(time.time() - start, 2), "ok"
        except FuturesTimeoutError:
            return None, round(time.time() - start, 2), "timeout"
        except Exception as e:
            return None, round(time.time() - start, 2), f"error:{e}"


def _raw_crewai():
    from agents.crewai_agents import crew
    return str(crew.kickoff())

def _raw_langgraph():
    import time as t
    from agents.langgraph_agents import build_graph, PortfolioState, TICKERS as LG_TICKERS
    app   = build_graph()
    state: PortfolioState = {
        "tickers": LG_TICKERS, "raw_data": {}, "analysis_table": "",
        "report": "", "start_time": t.time()
    }
    return app.invoke(state)["report"]

def _raw_openai_sdk():
    from agents.openai_sdk_agents import run_pipeline
    report, _ = asyncio.run(run_pipeline())
    return report

def _raw_smolagents():
    from agents.smolagents_agents import orchestrator
    task = (
        "You are a portfolio strategist for Nordic equity markets.\n"
        "Complete these steps in order:\n"
        f"1. Use stock_data_collector to fetch data for: {', '.join(TICKERS)}\n"
        "2. Use stock_analyzer to analyze the same tickers and get their metrics\n"
        "3. Generate a professional portfolio recommendation report that:\n"
        "   - Assigns BUY, HOLD, or AVOID to each of the 10 stocks\n"
        "   - Justifies each decision using RSI, Sharpe ratio, moving averages, volatility\n"
        "   - Ends with a clear summary table of all 10 stocks and their ratings"
    )
    return str(orchestrator.run(task))


_RUNNERS = {
    "CrewAI":     _raw_crewai,
    "LangGraph":  _raw_langgraph,
    "OpenAI SDK": _raw_openai_sdk,
    "Smolagents": _raw_smolagents,
}


def run_framework(name: str, historical: bool = False) -> tuple:
    """
    Run one framework pipeline and return (report, latency_label, used_fallback).

    historical=True sets BACKTEST_END_DATE in the environment so that
    tools/stock_data.py transparently fetches 6 months of data ending 3 months ago.
    """
    # Set / clear the backtest env var BEFORE importing or running anything
    if historical:
        os.environ["BACKTEST_END_DATE"] = BACKTEST_END_DATE
    else:
        os.environ.pop("BACKTEST_END_DATE", None)

    mode = "HISTORICAL" if historical else "CURRENT"
    print(f"\n{'='*70}")
    print(f"  [{name}]  {mode} run  {'(backtest date: ' + BACKTEST_END_DATE + ')' if historical else ''}")
    print("=" * 70)

    runner = _RUNNERS[name]

    if name == "Smolagents":
        result, elapsed, status = _run_with_timeout(runner, SMOLAGENTS_TIMEOUT)
    else:
        start = time.time()
        try:
            result  = runner()
            elapsed = round(time.time() - start, 2)
            status  = "ok"
        except Exception as e:
            result  = None
            elapsed = round(time.time() - start, 2)
            status  = f"error:{e}"

    os.environ.pop("BACKTEST_END_DATE", None)   # always clean up

    if result is None:
        tag = ">120s" if status == "timeout" else "error"
        print(f"  [{name}]  {tag} — loading fallback report from output/")
        report        = _load_fallback(name)
        latency_label = tag
        used_fallback = True
    else:
        report        = result
        latency_label = elapsed
        used_fallback = False

    return report, latency_label, used_fallback

# ══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if v is None:            return "n/a"
    if isinstance(v, bool):  return str(v)
    if isinstance(v, float): return f"{v:.2f}"
    return str(v)

def _fmt_pct(v) -> str:
    if v is None: return "n/a"
    return f"{v:+.2f}%"


def build_main_table(current: dict, backtest: dict, frameworks: list) -> str:
    col_w   = 14
    label_w = 36
    total_w = label_w + col_w * len(frameworks)
    sep     = "-" * total_w
    hsep    = "=" * total_w

    def row(label, *vals):
        return f"{label:<{label_w}}" + "".join(f"{str(v):>{col_w}}" for v in vals)

    def sys_val(f, key):
        return _fmt(current[f][key])

    def str_val(f, key):
        return _fmt(current[f]["structural"][key])

    def bt_val(f, key):
        return _fmt_pct(backtest[f][key])

    lines = [
        hsep,
        f"{'FRAMEWORK EVALUATION  —  ALL 3 LAYERS':^{total_w}}",
        hsep,
        row("Metric", *frameworks),
        sep,
        "  LAYER 1 — SYSTEM METRICS",
        sep,
        row("Latency (seconds)",          *[sys_val(f, "latency_seconds")  for f in frameworks]),
        row("Estimated Tokens",            *[sys_val(f, "estimated_tokens") for f in frameworks]),
        row("Report Complete (B/H/A)",     *[sys_val(f, "is_complete")      for f in frameworks]),
        sep,
        "  LAYER 2 — STRUCTURAL METRICS",
        sep,
        row("Stock Coverage (/10)",        *[str_val(f, "stock_coverage")   for f in frameworks]),
        row("Metric Mentions (/6)",        *[str_val(f, "metric_mentions")  for f in frameworks]),
        row("Word Count",                  *[str_val(f, "word_count")       for f in frameworks]),
        row("BUY Mentions",                *[str_val(f, "buy_count")        for f in frameworks]),
        row("HOLD Mentions",               *[str_val(f, "hold_count")       for f in frameworks]),
        row("AVOID Mentions",              *[str_val(f, "avoid_count")      for f in frameworks]),
        sep,
        f"  LAYER 3 — BACKTEST  ({BACKTEST_END_DATE} → {TODAY_STR})",
        sep,
        row("Avg BUY Return (%)",          *[bt_val(f, "avg_buy_return")    for f in frameworks]),
        row("Avg AVOID Return (%)",        *[bt_val(f, "avg_avoid_return")  for f in frameworks]),
        row("Backtest Score (%)",          *[bt_val(f, "backtest_score")    for f in frameworks]),
        hsep,
    ]
    return "\n".join(lines)


def build_backtest_detail(backtest: dict, actual_returns: dict, frameworks: list) -> str:
    w = 70
    lines = [
        "=" * w,
        "  BACKTEST DETAIL",
        f"  Period : {BACKTEST_END_DATE} → {TODAY_STR}",
        "=" * w,
        "",
        "  Actual Price Returns:",
    ]
    for ticker in TICKERS:
        ret = actual_returns.get(ticker)
        lines.append(f"    {ticker:<15}  {_fmt_pct(ret)}")

    lines.append("")

    for name in frameworks:
        bt           = backtest[name]
        fallback_tag = "  [fallback report — approx]" if bt.get("used_fallback") else ""
        buy_str      = ", ".join(bt.get("buy_tickers",   [])) or "none"
        avoid_str    = ", ".join(bt.get("avoid_tickers", [])) or "none"
        score        = bt.get("backtest_score")
        verdict      = ""
        if score is not None:
            verdict = "  ✓ BUY > AVOID" if score > 0 else "  ✗ AVOID > BUY"

        lines += [
            f"  {name}{fallback_tag}",
            f"    BUY picks         : {buy_str}",
            f"    AVOID picks       : {avoid_str}",
            f"    Avg BUY return    : {_fmt_pct(bt.get('avg_buy_return'))}",
            f"    Avg AVOID return  : {_fmt_pct(bt.get('avg_avoid_return'))}",
            f"    Backtest score    : {_fmt_pct(score)}{verdict}",
            "",
        ]

    scored = [(f, backtest[f]["backtest_score"]) for f in frameworks
              if backtest[f]["backtest_score"] is not None]
    if scored:
        winner, best_score = max(scored, key=lambda x: x[1])
        lines += [
            f"  Best backtest score: {winner}  ({_fmt_pct(best_score)})",
            "=" * w,
        ]

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation():
    frameworks = ["CrewAI", "LangGraph", "OpenAI SDK", "Smolagents"]

    os.environ.pop("BACKTEST_END_DATE", None)   # clean slate
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── PHASE 1: Current data  →  Layer 1 + 2 ────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1 — CURRENT EVALUATION  (Layers 1 & 2)")
    print("=" * 70)

    current_results = {}
    for name in frameworks:
        report, latency, fallback = run_framework(name, historical=False)
        current_results[name] = {
            "latency_seconds":  latency,
            "estimated_tokens": count_tokens_estimate(report),
            "is_complete":      check_completeness(report),
            "structural":       structural_metrics(report, TICKERS),
            "used_fallback":    fallback,
        }

    # ── PHASE 2: Historical data  →  Layer 3 (backtest) ──────────────────
    print("\n" + "=" * 70)
    print(f"  PHASE 2 — BACKTEST  (data ending {BACKTEST_END_DATE}, 6-month window)")
    print("=" * 70)

    backtest_reports = {}
    for name in frameworks:
        report, _, fallback = run_framework(name, historical=True)
        backtest_reports[name] = {"report": report, "used_fallback": fallback}

    # Fetch actual returns from backtest date to today
    actual_returns = fetch_actual_returns(TICKERS, BACKTEST_END_DATE)

    # Parse recommendations and score each framework
    backtest_results = {}
    print("\n  Parsing recommendations and scoring ...")
    for name, data in backtest_reports.items():
        recs = parse_recommendations(data["report"], TICKERS)
        bt   = score_backtest(recs, actual_returns)
        bt["recommendations"] = recs
        bt["used_fallback"]   = data["used_fallback"]
        backtest_results[name] = bt
        rated = {t: r for t, r in recs.items() if r}
        print(f"    {name}: {rated}")

    # ── OUTPUT ────────────────────────────────────────────────────────────
    main_table = build_main_table(current_results, backtest_results, frameworks)
    bt_detail  = build_backtest_detail(backtest_results, actual_returns, frameworks)

    print("\n\n" + main_table)
    print("\n" + bt_detail)

    table_path = os.path.join(OUTPUT_DIR, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(main_table + "\n\n" + bt_detail)
    print(f"\nComparison table  →  {table_path}")

    # Build JSON (no raw report text — keeps the file readable)
    json_out = {}
    for name in frameworks:
        cr = current_results[name]
        br = backtest_results[name]
        json_out[name] = {
            "layer1_system": {
                "latency_seconds":  cr["latency_seconds"],
                "estimated_tokens": cr["estimated_tokens"],
                "is_complete":      cr["is_complete"],
                "used_fallback":    cr["used_fallback"],
            },
            "layer2_structural": cr["structural"],
            "layer3_backtest": {
                "avg_buy_return":   br["avg_buy_return"],
                "avg_avoid_return": br["avg_avoid_return"],
                "backtest_score":   br["backtest_score"],
                "buy_tickers":      br["buy_tickers"],
                "hold_tickers":     br["hold_tickers"],
                "avoid_tickers":    br["avoid_tickers"],
                "recommendations":  br["recommendations"],
                "used_fallback":    br["used_fallback"],
            },
        }

    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Full JSON results →  {json_path}")


if __name__ == "__main__":
    run_evaluation()
