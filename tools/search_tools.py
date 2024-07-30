#%%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from llama_index.tools.tavily_research import TavilyToolSpec

def get_tavily_tool():
    """Returns the tavily search tool"""
    tavily_tool = TavilyToolSpec(api_key = os.getenv("TAVILY"))
    return tavily_tool.to_tool_list()
#%%
if __name__ == "__main__":
    import sys
    __curdir__ = os.getcwd()
    
    if "tools" in __curdir__:
        sys.path.append(os.path.join(
            __curdir__,
            "../src"
        ))
    else:
        sys.path.append("./src")
    from llamaindex_config import llm
    from llama_index.core.agent import ReActAgent
    
    tools = get_tavily_tool()
    agent = ReActAgent.from_tools(
        tools = tools,
        llm = llm
    )
    response = agent.chat(
        "What are Illumina's main challenges in 2024?"
    )
    print(str(response))
    
### Actual result from agent ###
#Based on the information gathered, Illumina's main challenges in 2024 appear to be:

#1. Developing, manufacturing, and launching new products and services amid the inherent challenges in these areas.
#2. Dealing with sluggish demand for its genetic testing tools and diagnostics products, which is expected to result in flat core revenue compared to 2023.
#3. Resolving ongoing ownership and operational challenges that are impacting the company's performance.

#The company will need to navigate these challenges in order to drive growth and profitability in 2024.

    
# %%
