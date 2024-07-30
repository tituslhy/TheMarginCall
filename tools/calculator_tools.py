#%%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from llama_index.tools.wolfram_alpha.base import WolframAlphaToolSpec

wolfram_api_key = os.getenv('WOLFRAM_API_KEY')

def get_calculator_tool():
    wolfram_tool = WolframAlphaToolSpec(app_id = wolfram_api_key)
    return wolfram_tool.to_tool_list()

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
    
    tools = get_calculator_tool()
    agent = ReActAgent.from_tools(
        tools = tools,
        llm = llm
    )
    response = agent.chat(
        "What is the mass of the helium in the sun and what is 100000 * 12312 * 123 + 123?"
    )
    print(str(response))
### Actual result from agent ###
#The mass of the helium in the sun is 4.002602 unified atomic mass units. 
#The result of 100000 * 12312 * 123 + 123 is 151437600123.