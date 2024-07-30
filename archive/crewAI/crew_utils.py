import os
from dotenv import load_dotenv, find_dotenv
import warnings

from crewai import Agent, Task, Process, Crew
from crewai_tools import LlamaIndexTool
from typing import List
from langchain_aws import ChatBedrockConverse
from langchain.agents import load_tools
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

from .crew_tasks import StockAnalysisTasks
from .crew_agents import (
    llm,
    data_analyst,
    technical_analyst,
    fundamental_analyst,
    research_analyst,
    professor,
    reporter,
    manager
)

_ = load_dotenv(find_dotenv())
warnings.filterwarnings('ignore')
__curdir__ = os.getcwd()

import sys

if ("crewAI" in __curdir__):
    sys.path.append(
        os.path.join(
            __curdir__,
            "../../tools"
        )
    )
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

## Define tools ##
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


## Define crew ##
tasks = StockAnalysisTasks()
agents = [
    fundamental_analyst,
    technical_analyst,
    research_analyst,
    reporter
]
class TheResearchCrew:
    """This crew undertakes research and writes a research report"""
    def __init__(self, 
                 agents = agents):
        self.agents = agents
    def kickoff(self, question):
        fa_task = tasks.fundamental_analysis(
            fundamental_analyst,
            question
        )
        ta_task = tasks.technical_analysis(
            technical_analyst,
            question
        )
        research_task = tasks.research(
            research_analyst,
            question,
        )
        report_task = tasks.report(
            reporter,
            question
        )
        self.crew = Crew(
            agents = agents,
            tasks = [
                fa_task,
                ta_task,
                research_task,
                report_task,
            ],
            manager_llm = llm,
            manager_agent = None,
            process = Process.hierarchical,
            verbose = True,
        )
        return self.crew.kickoff()