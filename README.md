An autonomous AI-powered stock portfolio analysis system built for the Helsinki Stock Exchange (OMXH).  
This project is part of a bachelor's thesis comparing two multi-agent AI frameworks — **LangGraph** and **CrewAI** — using GPT-4o as the primary LLM.

## What It Does

A multi-agent system where three specialized agents work together to:
1. **Fetch** real-time and historical stock data from OMXH via yfinance
2. **Analyze** key metrics — RSI, moving averages, Sharpe ratio, volatility
3. **Generate** a plain-language portfolio recommendation report

The same system is implemented in both LangGraph and CrewAI to compare performance, output quality, latency, and developer experience.

## Tech Stack

- **LLM:** GPT-4o (OpenAI)
- **Frameworks:** LangGraph, CrewAI
- **Data:** Yahoo Finance (yfinance)
- **Language:** Python 3.12


## Project Structure
omxh-portfolio-agent/
├── agents/       # Agent definitions for LangGraph and CrewAI
├── tools/        # Stock data fetching and analysis tools
├── data/         # Raw and processed stock data
├── output/       # Generated reports
├── main.py       # Entry point
└── .env          # API keys (not committed)


## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:

## Research Context

This project is the practical implementation component of a bachelor's thesis at  
**SAMK — Satakunta University of Applied Sciences**, Pori, Finland.  
The thesis evaluates LangGraph vs CrewAI across metrics including output quality, latency, token usage, and developer experience.

## Author

Davuthan Alataş — AI and Data Engineering, SAMK
EOF