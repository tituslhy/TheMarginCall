
import requests
from typing import List, Optional, Literal

from llama_index.core import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    AgentRunner,
    ReActAgent,
    StructuredPlannerAgent
)
from autogen.agentchat.contrib.llamaindex_conversable_agent import (
    LLamaIndexConversableAgent
)
from autogen import (
    UserProxyAgent, 
    ConversableAgent,
    Agent
)
import chainlit as cl
from typing import Union, Optional, Dict 
import warnings

warnings.filterwarnings("ignore")

# def chat_new_message(message, sender):
#     cl.run_sync(
#         cl.Message(
#             content="",
#             author=sender.name,
#         ).send()
#     )
#     content = message.get('content')
#     cl.run_sync(
#         cl.Message(
#             content=content,
#             author=sender.name,
#         ).send()
#     )

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

class ChainlitConversableAgent(ConversableAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        if isinstance(message, dict):
            message = message['content']
        cl.run_sync(
            cl.Message(
                content = f"{self.name} *Sending message to '{recipient.name}':*\n\n{message}",
                author=self.name,
            ).send()
        )
        super(ChainlitConversableAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitLLamaIndexConversableAgent(LLamaIndexConversableAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        if isinstance(message, dict):
            message = message['content']
        cl.run_sync(
            cl.Message(
                content = f"{self.name} *Sending message to '{recipient.name}':*\n\n{message}",
                author=self.name,
            ).send()
        )
        super(ChainlitLLamaIndexConversableAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitUserProxyAgent(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to chat_manager. Press enter to skip and use auto-reply"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="continue", label="✅ Continue"
                        ),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        cl.Action( 
                            name="exit",
                            value="exit", 
                            label="🔚 Exit Conversation" 
                        ),
                    ],
                )
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=60))
        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if isinstance(message, dict):
            message = message['content']
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                author="UserProxyAgent",
            ).send()
        )
        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

def get_agent(llm,
              tools: List[BaseToolSpec],
              agent_name: str,
              agent_description: str,
              system_message: Optional[str] = None,
              human_input_mode: Optional[str] = "NEVER",
              agent_type: Optional[Literal[
                  "function",
                  "react",
                  "structured"
              ]] = "function"
              ):
    if agent_type == "react":
        return ChainlitLLamaIndexConversableAgent(
            name = agent_name,
            llama_index_agent = ReActAgent.from_tools(
                tools,
                llm = llm,
                verbose = True,
            ),
            system_message = system_message,
            description = agent_description,
            human_input_mode = human_input_mode,
            max_consecutive_auto_reply = 10,
        )
    else:
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools = tools,
            llm = llm,
            verbose = True
        )
        if agent_type == "structured":
            return ChainlitLLamaIndexConversableAgent(
                name = agent_name,
                llama_index_agent = StructuredPlannerAgent(
                    agent_worker = agent_worker,
                    tools = tools,
                    verbose = True,
                ),
                system_message = system_message,
                description = agent_description,
                human_input_mode = human_input_mode,
                max_consecutive_auto_reply = 10,
            )
        return ChainlitLLamaIndexConversableAgent(
            name = agent_name,
            llama_index_agent = agent_worker.as_agent(),
            system_message = system_message,
            description = agent_description,
            human_input_mode = human_input_mode,
            max_consecutive_auto_reply = 10,
        )
    