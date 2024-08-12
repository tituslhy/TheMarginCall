#%%
from .autogen_utils import get_agent, ChainlitUserProxyAgent

import os
import sys

__curdir__ = os.getcwd()
if "src" in __curdir__:
    sys.path.append("../tools")
else:
    sys.path.append("./tools")
    
from calculator_tools import get_calculator_tool
from data_analysis_tools import get_da_tools
from fundamental_analysis_tools import get_fa_tools
from rag_tools import get_rag_tools
from search_tools import get_tavily_tool
from sec_tools import get_sec_tool
from technical_analysis_tools import get_ta_tools  
from gmail_tool import get_gmail_tool

import autogen

from llama_index.core import Settings
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding

#%%
# llm_config_list = [
#     {
#         "model": "gemini-1.5-flash",
#         "api_type": "google"
#     }
# ]

# seed=25

# llm_config={"config_list": llm_config_list, "seed": seed}

llm_config = {
    "model": os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_type": "azure",
    "api_version": os.environ["AZURE_API_VERSION"]
}

calculator_tool = get_calculator_tool()
da_tool = get_da_tools()
fa_tool = get_fa_tools()
textbook_tool = get_rag_tools()
search_tool = get_tavily_tool()
sec_tool = get_sec_tool()
ta_tool = get_ta_tools()
gmail_tool = get_gmail_tool()

Settings.llm = BedrockConverse(
    model = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name = "us-east-1",
    max_tokens=4000,
)
Settings.embed_model = BedrockEmbedding(
    model = "amazon.titan-embed-text-v1",
    aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_region_name = os.environ["AWS_DEFAULT_REGION"]
)

def get_groupchat(llm_config = llm_config):
    """Instantiates group chat object. Returns user_proxy agent and
    manager agent"""
    user_proxy = ChainlitUserProxyAgent(
        name="Admin",
        human_input_mode="ALWAYS",
        code_execution_config=False
    )
    
    data_analyst = get_agent(
        llm = Settings.llm,
        agent_name = "Principal_data_analyst",
        tools = [*da_tool],
        system_message = """You are an expert in statistics and helps customers
        develop data-driven insights from data analysis using statistical tools and
        methods to guide decision-making.""",
        agent_description = """This agent helps customers undertake statistical analysis
        of financial market data using methods such as correlations, compounded annual
        growth rate, etc.""",
        agent_type = "react",
    )

    technical_analyst = get_agent(
        llm = Settings.llm,
        agent_name = "Principal_technical_analyst",
        tools = [*ta_tool],
        system_message = """You are the top technical analyst of the field, adroit
        at crystallizing insights and ivnestment strategies from stock data. Use tools
        to compute important technical analysis metrics to guide your investment 
        recommendations.""",
        agent_description="""This agent helps customers undertake technical analysis of
        financial market data using methods such as stochastic relative strength index,
        bollinger bands, ichimoku cloud, etc.""",
        agent_type = "function",
    )

    fundamental_analyst = get_agent(
        agent_name="Principal_fundamental_analyst",
        tools = [fa_tool],
        system_message = """You are the top fundamental analst of the field, adroit
        at crystallizing insights and investment strategies from stock data.""",
        agent_description="""This agent helps customers undertake fundamental analysis
        of companies by looking at financial market data.""",
        llm = Settings.llm,
        agent_type = "function",
    )

    research_analyst = get_agent(
        agent_name="Principal_researcher",
        tools = [
            *search_tool,
            *sec_tool,
            *calculator_tool,
        ],
        system_message = """You are the top finance researcher of the field, adroit
        at crystallizing insights and investment strategies from
        close reading of SEC reports and research articles online.""",
        agent_description="""This agent helps customers undertake analysis of the 
        financial performance of companies by close reading of SEC reports and 
        research articles online.""",
        llm = Settings.llm,
        agent_type = "react",
    )

    professor = get_agent(
        agent_name="Distinguished_professor_of_finance",
        tools = [
            *textbook_tool,
        ],
        system_message="""You are a distinguished professor of Finance with a 
        specialization in investment finance. Your role is to answer specific
        questions on finance and review the recommendations made by other agents.
        Always provide a layman understanding of the metrics reported - for example
        if the technical analysts says that the Aroon indicator shows a buy signal,
        first explain what the indicator does and the interpretation of a buy signal.
        Taking the signals from all metrics in totality, make a recommendation -
        should the investor buy/sell/do more research?""",
        agent_description="""Answers questions on technical definitions, concepts and 
        general questions on how to get started on investments. This agent also reviews
        the recommendations made by analysts
        """,
        llm = Settings.llm,
        agent_type = "react",
    )

    reporter = get_agent(
        agent_name="Principal_finance_reporter",
        tools = [
            *gmail_tool,
        ],
        system_message="""You are a Pulitzer prize winning reporter adroit at 
        distilling complex concepts to crystal clear insights easily understood by 
        the layperson. You endeavor to help customers understand the investment 
        recommendations put forth by your team. Only access the user's emails if
        the user gives express permission.""",
        agent_description = """Use this agent to write a report, draft emails and
        send emails.""",
        llm = Settings.llm
    )

    agents = [
        technical_analyst,
        data_analyst,
        fundamental_analyst,
        professor,
        reporter,
        user_proxy,
        research_analyst
    ]

    groupchat = autogen.GroupChat(
        agents = agents,
        messages = [],
        max_round = 5000,
        allowed_or_disallowed_speaker_transitions = {
            user_proxy: [
                technical_analyst, 
                fundamental_analyst,
                data_analyst,
                research_analyst,
                professor,
                reporter,
            ],
            technical_analyst:[professor],
            fundamental_analyst: [professor],
            data_analyst: [user_proxy],
            research_analyst: [user_proxy],
            professor: [user_proxy],
            reporter:[user_proxy],
        },
        enable_clear_history = False,
        speaker_transitions_type="allowed",
        select_speaker_message_template="""
        You spearhead an investment crew known as The Margin Call. 
        
        Route all investment analysis related questions to either the principal 
        technical analyst or the principal fundamental analyst. Once either 
        analyst is done with their tasks, they'll route it to the distinguished 
        professor of finance for review.
        
        Use the data analyst for all forecasting tasks.
        
        Route all investment concept related questions to the distinguished 
        professor of finance directly. 
        
        Route all requests for report writing to the principal finance reporter.
        Always check these requests to see if there is a need to write and send
        an email. If there is, check that you have received a destination email
        address.
        
        Route everything else to the principal research analyst.
        """
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config = llm_config
    )
    
    return user_proxy, manager, groupchat