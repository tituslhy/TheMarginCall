#%%
import os
import sys

__curdir__ = os.getcwd()
if "src" in __curdir__:
    sys.path.append("../../tools")
    prompt_path = "../../tools/prompt.txt"
else:
    sys.path.append("./tools")
    prompt_path = "./tools/prompt.txt"
    
from calculator_tools import get_calculator_tool
from data_analysis_tools import get_da_tools
from fundamental_analysis_tools import get_fa_tools
from rag_tools import get_rag_tools
from search_tools import get_tavily_tool
from sec_tools import get_sec_tool
from technical_analysis_tools import get_ta_tools  
from gmail_tool import get_gmail_tool

from crewai_tools import LlamaIndexTool, tool

from typing import List

import chainlit as cl
from chainlit import run_sync

def calculator_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_calculator_tool()]
def da_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_da_tools()]
def textbook_tool(prompt_path: str = prompt_path) -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_rag_tools(
        prompt_path=prompt_path
    )]
def fa_tool() -> LlamaIndexTool:
    return LlamaIndexTool.from_tool(get_fa_tools())
def ta_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_ta_tools()]
def search_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_tavily_tool()]
def sec_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_sec_tool()]
def gmail_tool() -> List[LlamaIndexTool]:
    return [LlamaIndexTool.from_tool(t) for t in get_gmail_tool()]

@tool("Ask human follow up questions")
def ask_human(question: str) -> str:
    """Ask human for feedback or follow up questions"""
    human_response = run_sync(
        cl.AskUserMessage(content=f'{question}').send()
    )
    if human_response:
        return human_response['output']
    
#%%
if __name__ == "__main__":
    
    print(calculator_tool())
    print(da_tool())
    print(fa_tool())
    print(textbook_tool())

# %%
