import time

from langchain import OpenAI, Wikipedia
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.callbacks import get_openai_callback

from utils.CustomDocstoreExplorer import CustomDocstoreExplorer


class ReactBase:
    def __init__(self, model_name="text-davinci-003", verbose=True):
        self.model_name = model_name
        self.verbose = verbose
        self.tools = self._load_tools()
        self.agent = initialize_agent(self.tools,
                                      OpenAI(temperature=0, model_name=self.model_name),
                                      agent=AgentType.REACT_DOCSTORE,
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
            result["total_tokens"] = cb.total_tokens
            result["prompt_tokens"] = cb.prompt_tokens
            result["completion_tokens"] = cb.completion_tokens
            result["total_cost"] = cb.total_cost
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
        self.agent = initialize_agent(self.tools,
                                      OpenAI(temperature=0, model_name=self.model_name),
                                      agent=AgentType.REACT_DOCSTORE,
                                      verbose=self.verbose,
                                      return_intermediate_steps=True)

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
