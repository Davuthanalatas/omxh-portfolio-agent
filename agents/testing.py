import yfinance as yf
from langchain_core.tools import tool

@tool
def get_stock_info(ticker: str) -> str:
    """Fetches basic stock information for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    info = stock.info

    return f"""
    Company: {info.get('longName', 'N/A')}
    Sector: {info.get('sector', 'N/A')}
    Current Price: {info.get('currentPrice', 'N/A')}
    52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
    52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
    Market Cap: {info.get('marketCap', 'N/A')}
    P/E Ratio: {info.get('trailingPE', 'N/A')}
    Analyst Recommendation: {info.get('recommendationKey', 'N/A')}
    """
@tool
def get_stock_news(ticker: str) -> str:
    """Fetches the latest news headlines for a given stock ticker."""
    stock = yf.Ticker(ticker)
    news = stock.news

    if not news:
        return "No recent news found."

    headlines = []
    for item in news[:5]:  # top 5 articles
        content = item.get("content", {})
        title = content.get("title", "No title")
        headlines.append(f"- {title}")

    return "\n".join(headlines)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

load_dotenv()

# 1. The LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. The tools list
tools = [get_stock_info, get_stock_news]

# 3. Create the agent
agent = create_agent(llm, tools)

# 4. Run it
# result = agent.invoke({
#     "messages": [("user", "Analyze NOKIA.HE for me and give me a summary.")]
# })

# # 5. Print the final response
# print(result["messages"][-1].content)

result = agent.invoke({
    "messages": [("user", "Get me stock info and latest news for NOKIA.HE")]
})
print(result["messages"][-1].content)