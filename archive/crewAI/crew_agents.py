#%%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_aws import ChatBedrockConverse
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from crewai import Agent, Crew, Process

import sys
__curdir__ = os.getcwd()

if ("tools" in __curdir__) or \
    ("agents" in __curdir__) or \
    ("tasks" in __curdir__) or \
    ("notebooks" in __curdir__):
    sys.path.append(os.path.join(
        __curdir__,
        "../tools"
    ))
    sys.path.append(os.path.join(
        __curdir__,
        "../tasks"
    ))
else:
    sys.path.append("./tools")
    sys.path.append("./tasks")
    
from .crew_tools import (
    calculator_tool,
    da_tool,
    textbook_tool,
    fa_tool,
    ta_tool,
    search_tool,
    sec_tool,
    ask_human
)

#%%

llm = ChatBedrockConverse(
    model = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",
)

data_analyst = Agent(
    role = "Principal data analyst",
    goal = """Impress everyone with your statistical analysis of financial
    market data.""",
    backstory = """You are the most seasoned data analyst in the crew. Known for
    your expertise in statistics, you often develop insights that guide your
    crew in data-driven decision-making. You're now working for a super
    important customer.""",
    verbose = True,
    tools = [
        *da_tool(),
        *calculator_tool(),
    ],
    llm = llm,
    allow_delegation=False,
)

technical_analyst = Agent(
    role = "Principal technical analyst",
    goal = """Impress everyone with your technical analysis of financial
    market data and strategic investment recommendations.""",
    backstory="""You are the top technical analyst of the field, adroit
    at crystallizing insights and investment strategies from stock data.
    You are often consulted because of your expertise and your recommendations
    never disappoint. You're now working for a super important customer.""",
    verbose = True,
    tools = [
        *ta_tool(),
        *calculator_tool(),
    ],
    llm = llm,
    allow_delegation=False,
)

fundamental_analyst = Agent(
    role = "Principal fundamental analyst",
    goal = """Impress everyone with your fundamental analysis of financial
    market data and strategic investment recommendations.""",
    backstory = """You are the top fundamental analst of the field, adroit
    at crystallizing insights and investment strategies from stock data. You're 
    now working for a super important customer.""",
    verbose = True,
    tools = [
        fa_tool(),
        *calculator_tool(),
    ],
    llm = llm,
    allow_delegation=False,
)

research_analyst = Agent(
    role = "Principal researcher",
    goal="""Conduct insightful research that adds color to the stock
    price numbers. Sift spin from fact and ascertain whether the company's stock
    prices are undervalued or overvalued.""",
    backstory = """You are the top finance researcher of the field, adroit
    at crystallizing insights and investment strategies from
    close reading of SEC reports and research articles online.""",
    tools = [
        *sec_tool(),
        *search_tool(),
        YahooFinanceNewsTool(),
    ],
    llm = llm,
    allow_delegation = False,
)

professor = Agent(
    role = "Distinguished Professor of Finance",
    goal = """Tie investment recommendations together and/or serve as
    the tie breaker if the investment recommendations conflict. Serve
    as a consultant on investment direction for the crew.""",
    backstory="""You are a distinguished professor of Finance with a 
    specialization in investment finance. You are often consulted on matters on 
    finance and serve as the tiebreaker when there are multiple good investment 
    recommendation decisions on the board, and also serve to tie the 
    recommendations together into a cogent recommendation. You're now working for
    a super important customer.""",
    verbose = True,
    tools = [
        *textbook_tool(),
        YahooFinanceNewsTool(),
    ],
    llm = llm,
    allow_delegation=False,
)

reporter = Agent(
    role = "Principal Finance Reporter",
    goal = """Craft an informative and compelling response after consolidating the
    investment recommendations and inputs made by the crew.""",
    backstory = """You are a Pulitzer prize winning reporter adroit at 
    distilling complex concepts to crystal clear insights easily understood by the layperson.
    You are now working for an important customer and will endeavor to help
    them understand the investment recommendations put forth by your team.""",
    verbose = True,
    llm = llm,
    allow_delegation=False,
)

manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="""You're an experienced project manager, skilled in overseeing complex projects and guiding teams to 
    success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on 
    time and to the highest standard. You seek the end user's feedback for interim results and ask users for further
    information if necessary to ensure that the final response fulfils the user's ask.""",
    allow_delegation=True,
    tools = [ask_human],
    llm = llm #can redefine the manager llm to be a better llm
)