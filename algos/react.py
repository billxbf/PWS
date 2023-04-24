import time

from langchain import OpenAI, Wikipedia
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.callbacks import get_openai_callback
from utils.util import *
from langchain.chat_models import ChatOpenAI
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.tools.base import BaseTool
from langchain.prompts.base import BasePromptTemplate
from typing import Any, List, Optional, Sequence, Tuple
from prompts.wiki_prompt import *

from utils.CustomDocstoreExplorer import CustomDocstoreExplorer


class ReactBase:
    def __init__(self, fewshot, model_name="text-davinci-002", max_iter=8, verbose=True):
        self.model_name = model_name
        self.max_iter = max_iter
        self.verbose = verbose
        self.fewshot = fewshot
        self.tools = self._load_tools()
        if model_name in OPENAI_COMPLETION_MODELS:
            self.agent = initialize_agent(self.tools,
                                          OpenAI(temperature=0, model_name=self.model_name),
                                          agent=AgentType.REACT_DOCSTORE,
                                          verbose=self.verbose,
                                          return_intermediate_steps=True,
                                          max_iterations=max_iter)
        elif model_name in OPENAI_CHAT_MODELS:
            self.agent = initialize_agent(self.tools,
                                          ChatOpenAI(temperature=0, model_name=self.model_name),
                                          agent=AgentType.REACT_DOCSTORE,
                                          verbose=self.verbose,
                                          return_intermediate_steps=True,
                                          max_iterations=max_iter)
        self.agent.agent.llm_chain.prompt.template = fewshot

    def run(self, prompt):
        self.reset()
        result = {}
        with get_openai_callback() as cb:
            st = time.time()
            response = self.agent(prompt)
            result["wall_time"] = time.time() - st
            result["input"] = response["input"]
            result["output"] = response["output"]
            result["intermediate_steps"] = response["intermediate_steps"]
            result["tool_usage"] = self._parse_tool(response["intermediate_steps"])
            result["total_tokens"] = cb.total_tokens
            result["prompt_tokens"] = cb.prompt_tokens
            result["completion_tokens"] = cb.completion_tokens
            result["total_cost"] = cb.total_cost
            result["steps"] = len(response["intermediate_steps"]) + 1
            result["token_cost"] = result["total_cost"]
            result["tool_cost"] = 0
        return result

    def _load_tools(self):
        docstore = CustomDocstoreExplorer(Wikipedia())
        return [
            Tool(
                name="Search",
                func=docstore.search,
                description="useful for when you need to ask with search"
            ),
            Tool(
                name="Lookup",
                func=docstore.lookup,
                description="useful for when you need to ask with lookup"
            )
        ]

    def reset(self):
        self.tools = self._load_tools()
        if self.model_name in OPENAI_COMPLETION_MODELS:
            self.agent = initialize_agent(self.tools,
                                          OpenAI(temperature=0, model_name=self.model_name),
                                          agent=AgentType.REACT_DOCSTORE,
                                          verbose=self.verbose,
                                          return_intermediate_steps=True,
                                          max_iterations=self.max_iter)
        elif self.model_name in OPENAI_CHAT_MODELS:
            self.agent = initialize_agent(self.tools,
                                          ChatOpenAI(temperature=0, model_name=self.model_name),
                                          agent=AgentType.REACT_DOCSTORE,
                                          verbose=self.verbose,
                                          return_intermediate_steps=True,
                                          max_iterations=self.max_iter)
        self.agent.agent.llm_chain.prompt.template = self.fewshot

    def _parse_tool(self, intermediate_steps):
        tool_usage = {"search": 0, "lookup": 0}
        for step in intermediate_steps:
            if step[0].tool == "Search":
                tool_usage["search"] += 1
            if step[0].tool == "Lookup":
                tool_usage["lookup"] += 1
        return tool_usage


# React with zero-shot prompting; Instead of Wikipedia search and lookup, ReactZeroShot uses google search
# and a LLM based calculator (LLM+Python REPL)
class ReactZeroShot(ReactBase):
    def __init__(self, model_name="text-davinci-003", verbose=True):
        self.model_name = model_name
        self.verbose = verbose

        self.tools = self._load_tools()
        self.agent = initialize_agent(self.tools,
                                      OpenAI(temperature=0, model_name=self.model_name),
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose=self.verbose,
                                      return_intermediate_steps=True)

    def run(self, prompt):
        self.reset()
        result = {}
        with get_openai_callback() as cb:
            st = time.time()
            response = self.agent(prompt)
            result["wall_time"] = time.time() - st
            result["input"] = response["input"]
            result["output"] = response["output"]
            result["intermediate_steps"] = response["intermediate_steps"]
            result["tool_usage"] = self._parse_tool(response["intermediate_steps"])
            result["total_tokens"] = cb.total_tokens + result["tool_usage"]["llm-math_token"]
            result["prompt_tokens"] = cb.prompt_tokens
            result["completion_tokens"] = cb.completion_tokens
            result["total_cost"] = cb.total_cost + result["tool_usage"]["llm-math_token"] * 0.000002 + \
                                   result["tool_usage"]["serpapi"] * 0.01  # Developer Plan

        return result

    def _load_tools(self):
        return load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0, model_name=self.model_name))

    def reset(self):
        self.tools = self._load_tools()
        self.agent = initialize_agent(self.tools,
                                      OpenAI(temperature=0, model_name=self.model_name),
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose=self.verbose,
                                      return_intermediate_steps=True)

    def _parse_tool(self, intermediate_steps):
        tool_usage = {"serpapi": 0, "llm-math_token": 0}
        for step in intermediate_steps:
            if step[0].tool == "Search":
                tool_usage["serpapi"] += 1
            if step[0].tool == "Calculator":
                tool_usage["llm-math_token"] += len(step[0].tool_input + step[1]) // 4  # 4 chars per token
        return tool_usage
