import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ── LATENCY ───────────────────────────────────────────────────────────────
# We wrap each system's run in a timer to measure wall-clock time.

def measure_latency(run_func) -> tuple[any, float]:
    start = time.time()
    result = run_func()
    elapsed = round(time.time() - start, 2)
    return result, elapsed

# ── TOKEN USAGE ───────────────────────────────────────────────────────────
# We ask GPT-4o to evaluate the report AND return token counts.
# For CrewAI we estimate tokens since it doesn't expose them directly.
# For LangGraph we can track them precisely since we call OpenAI ourselves.

def count_tokens_estimate(text: str) -> int:
    # Rough estimate: 1 token ≈ 4 characters (OpenAI rule of thumb)
    return len(text) // 4

# ── OUTPUT QUALITY SCORING ────────────────────────────────────────────────
# We use GPT-4o as an independent judge to score each report.
# This is called LLM-as-a-Judge — a standard evaluation technique in NLP research.
# The judge scores on 3 dimensions, each out of 10.

def evaluate_report_quality(report: str, framework: str) -> dict:
    print(f"\nEvaluating {framework} report quality...")

    prompt = f"""You are an objective evaluator of AI-generated financial reports.

Score the following portfolio recommendation report on three dimensions.
Return ONLY a valid JSON object with no extra text.

Scoring dimensions (each 0-10):
1. structure_score: Is the report well-organized with clear BUY/HOLD/AVOID sections?
2. justification_score: Are recommendations backed by specific metrics (RSI, Sharpe, etc.)?
3. clarity_score: Is the language clear, professional, and easy to understand?

Also provide:
- total_score: average of the three scores
- strengths: one sentence on what the report does well
- weaknesses: one sentence on what could be improved

Report to evaluate:
{report}

Return only this JSON structure:
{{
    "structure_score": <0-10>,
    "justification_score": <0-10>,
    "clarity_score": <0-10>,
    "total_score": <0-10>,
    "strengths": "<one sentence>",
    "weaknesses": "<one sentence>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    scores = json.loads(raw)
    return scores

# ── BUY/HOLD/AVOID DETECTION ──────────────────────────────────────────────
# Checks whether the report actually contains the required recommendation structure.
# A report that doesn't mention BUY/HOLD/AVOID is considered unreliable.

def check_structure_reliability(report: str) -> dict:
    report_upper = report.upper()
    return {
        "has_buy": "BUY" in report_upper,
        "has_hold": "HOLD" in report_upper,
        "has_avoid": "AVOID" in report_upper,
        "is_complete": all(k in report_upper for k in ["BUY", "HOLD", "AVOID"])
    }

# ── FULL EVALUATION ───────────────────────────────────────────────────────
# Runs both systems, measures everything, and prints a comparison table.

def run_evaluation():
    from agents.crewai_agents import crew, TICKERS
    from agents.langgraph_agents import build_graph, PortfolioState, TICKERS as LG_TICKERS

    results = {}

    # --- CrewAI ---
    print("=" * 60)
    print("Running CrewAI system...")
    print("=" * 60)

    def run_crewai():
        return crew.kickoff()

    crewai_output, crewai_latency = measure_latency(run_crewai)
    crewai_report = str(crewai_output)

    crewai_tokens = count_tokens_estimate(crewai_report)
    crewai_quality = evaluate_report_quality(crewai_report, "CrewAI")
    crewai_reliability = check_structure_reliability(crewai_report)

    results["CrewAI"] = {
        "latency_seconds": crewai_latency,
        "estimated_tokens": crewai_tokens,
        "quality": crewai_quality,
        "reliability": crewai_reliability
    }

    # --- LangGraph ---
    print("\n" + "=" * 60)
    print("Running LangGraph system...")
    print("=" * 60)

    def run_langgraph():
        import time as t
        app = build_graph()
        initial_state: PortfolioState = {
            "tickers": LG_TICKERS,
            "raw_data": {},
            "analysis_table": "",
            "report": "",
            "start_time": t.time()
        }
        return app.invoke(initial_state)

    lg_output, lg_latency = measure_latency(run_langgraph)
    lg_report = lg_output["report"]

    lg_tokens = count_tokens_estimate(lg_report)
    lg_quality = evaluate_report_quality(lg_report, "LangGraph")
    lg_reliability = check_structure_reliability(lg_report)

    results["LangGraph"] = {
        "latency_seconds": lg_latency,
        "estimated_tokens": lg_tokens,
        "quality": lg_quality,
        "reliability": lg_reliability
    }

    # --- Print Comparison Table ---
    print("\n")
    print("=" * 60)
    print("        EVALUATION RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'CrewAI':>12} {'LangGraph':>12}")
    print("-" * 60)
    print(f"{'Latency (seconds)':<30} {crewai_latency:>12} {lg_latency:>12}")
    print(f"{'Estimated Tokens':<30} {crewai_tokens:>12} {lg_tokens:>12}")
    print(f"{'Structure Score (0-10)':<30} {crewai_quality['structure_score']:>12} {lg_quality['structure_score']:>12}")
    print(f"{'Justification Score (0-10)':<30} {crewai_quality['justification_score']:>12} {lg_quality['justification_score']:>12}")
    print(f"{'Clarity Score (0-10)':<30} {crewai_quality['clarity_score']:>12} {lg_quality['clarity_score']:>12}")
    print(f"{'Total Quality Score (0-10)':<30} {crewai_quality['total_score']:>12} {lg_quality['total_score']:>12}")
    print(f"{'Report Complete (BUY/HOLD/AVOID)':<30} {str(crewai_reliability['is_complete']):>12} {str(lg_reliability['is_complete']):>12}")
    print("=" * 60)

    print(f"\nCrewAI  → Strengths: {crewai_quality['strengths']}")
    print(f"           Weaknesses: {crewai_quality['weaknesses']}")
    print(f"\nLangGraph → Strengths: {lg_quality['strengths']}")
    print(f"             Weaknesses: {lg_quality['weaknesses']}")

    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to output/evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()