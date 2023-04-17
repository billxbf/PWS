# Workers who take query from Planner and execute them leveraging different Tools.
from langchain import OpenAI, LLMMathChain, LLMChain, PromptTemplate, Wikipedia
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer
from langchain.utilities import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from nodes.Node import Node


class GoogleWorker(Node):
    def __init__(self, name="Google"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that searches results from Google. Useful when you need to find short " \
                           "and general answers about a specific topic. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        tool = SerpAPIWrapper()
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class WikipediaWorker(Node):
    def __init__(self, name="Wikipedia", docstore=None):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that search for similar page contents from Wikipedia. Useful when you need to " \
                           "get holistic knowledge about people, places, companies, historical events, " \
                           "or other subjects. The response are long and might contain some irrelevant information. " \
                           "Input should be a search query."
        self.docstore = docstore

    def run(self, input, log=False):
        if not self.docstore:
            self.docstore = DocstoreExplorer(Wikipedia())
        assert isinstance(input, self.input_type)
        tool = Tool(
            name="Search",
            func=self.docstore.search,
            description="useful for when you need to ask with search"
        )
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class DocStoreLookUpWorker(Node):
    def __init__(self, name="LookUp", docstore=None):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that search the direct sentence in current Wikipedia result page. Useful when you " \
                           "need to find information about a specific keyword from a existing Wikipedia search " \
                           "result. Input should be a search keyword."
        self.docstore = docstore

    def run(self, input, log=False):
        if not self.docstore:
            raise ValueError("Docstore must be provided for lookup")
        assert isinstance(input, self.input_type)
        tool = Tool(
            name="Lookup",
            func=self.docstore.lookup,
            description="useful for when you need to ask with lookup"
        )
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class CustomWolframAlphaAPITool(WolframAlphaAPIWrapper):
    def __init__(self):
        super().__init__()

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse result."""
        res = self.wolfram_client.query(query)

        try:
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it"

        if answer is None or answer == "":
            return "No good Wolfram Alpha Result was found"
        else:
            return f"Answer: {answer}"


class WolframAlphaWorker(Node):
    def __init__(self, name="WolframAlpha"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "A wolfram alpha search engine. Useful when you need short answers about Math, " \
                           "Science, Technology. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        tool = CustomWolframAlphaAPITool()
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class CalculatorWorker(Node):
    def __init__(self, name="Calculator"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.description = "A calculator that can compute arithmetic expressions. Useful when you need to perform " \
                           "math calculations. Input should be a mathematical expression"

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        llm = OpenAI(temperature=0)
        tool = LLMMathChain(llm=llm, verbose=False)
        response = tool(input)
        evidence = response["answer"]
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["question"], "output": response["answer"]}
        return evidence


class LLMWorker(Node):
    def __init__(self, name="LLM"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.description = "A pretrained LLM like yourself. Useful when you need to act with general world " \
                           "knowledge and common sense. Prioritize it when you are confident in solving the problem " \
                           "yourself. Input can be any instruction."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        llm = OpenAI(temperature=0)
        prompt = PromptTemplate(template="Respond directly with no extra words.\n\n{request}", input_variables=["request"])
        tool = LLMChain(prompt=prompt, llm=llm, verbose=False)
        response = tool(input)
        evidence = response["text"].strip("\n")
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["request"], "output": response["text"]}
        return evidence


WORKER_REGISTRY = {"Google": GoogleWorker(),
                   "Wikipedia": WikipediaWorker(),
                   "LookUp": DocStoreLookUpWorker(),
                   "WolframAlpha": WolframAlphaWorker(),
                   "Calculator": CalculatorWorker(),
                   "LLM": LLMWorker()}
