# Workers who take query from Planner and execute them leveraging different Tools.
from langchain import OpenAI, LLMMathChain, LLMChain, PromptTemplate, Wikipedia
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer
from langchain.utilities import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import requests
from geopy.geocoders import Nominatim

from nodes.Node import Node


class GoogleWorker(Node):
    def __init__(self, name="Google"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that searches results from Google. Useful when you need to find short " \
                           "and succinct answers about a specific topic. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        tool = SerpAPIWrapper()
        evidence = tool.run(input)
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class YelpWorker(Node):
    def __init__(self, name="Yelp"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "Worker that searches results from Yelp. Useful when you need to find reviews about a " \
                           "restaurant. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        tool = SerpAPIWrapper()
        tool.params = {
            "engine": "yelp",
        }
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
        self.description = "A WolframAlpha search engine. Useful when you need to solve a complicated Mathematical or " \
                           "Algebraic equation. Input should be an equation or function."

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
        prompt = PromptTemplate(template="Respond in short directly with no extra words.\n\n{request}",
                                input_variables=["request"])
        tool = LLMChain(prompt=prompt, llm=llm, verbose=False)
        response = tool(input)
        evidence = response["text"].strip("\n")
        assert isinstance(evidence, self.output_type)
        if log:
            return {"input": response["request"], "output": response["text"]}
        return evidence


class ZipCodeRetriever(Node):

    def __init__(self, name="ZipCodeRetriever"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = False
        self.description = "A zip code retriever. Useful when you need to get users' current zip code. Input can be " \
                           "left blank."

    def get_ip_address(self):
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data["ip"]

    def get_location_data(sefl,ip_address):
        url = f"https://ipinfo.io/{ip_address}/json"
        response = requests.get(url)
        data = response.json()
        return data

    def get_zipcode_from_lat_long(self, lat, long):
        geolocator = Nominatim(user_agent="zipcode_locator")
        location = geolocator.reverse((lat, long))
        return location.raw["address"]["postcode"]

    def get_current_zipcode(self):
        ip_address = self.get_ip_address()
        location_data = self.get_location_data(ip_address)
        lat, long = location_data["loc"].split(",")
        zipcode = self.get_zipcode_from_lat_long(float(lat), float(long))
        return zipcode

    def run(self, input):
        assert isinstance(input, self.input_type)
        evidence = self.get_current_zipcode()
        assert isinstance(evidence, self.output_type)


class SearchDocWorker(Node):

    def __init__(self, doc_name, doc_path, name="SearchDoc"):
        super().__init__(name, input_type=str, output_type=str)
        self.isLLMBased = True
        self.doc_path = doc_path
        self.description = f"A vector store that searches for similar and related content in document: {doc_name}. " \
                           f"The result might contain some irrelevant information in response chunk so always use an LLM " \
                           f"afterwards to retrieve key knowledge. Input should be a search query."

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        loader = TextLoader(self.doc_path)
        vectorstore = VectorstoreIndexCreator().from_loaders([loader]).vectorstore
        evidence = vectorstore.similarity_search(input, k=1)[0].page_content
        assert isinstance(evidence, self.output_type)
        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class SearchSOTUWorker(SearchDocWorker):
    def __init__(self, name="SearchSOTU"):
        super().__init__(name=name, doc_name="state_of_the_union", doc_path="data/docs/state_of_the_union.txt")


WORKER_REGISTRY = {"Google": GoogleWorker(),
                   "Wikipedia": WikipediaWorker(),
                   "LookUp": DocStoreLookUpWorker(),
                   "WolframAlpha": WolframAlphaWorker(),
                   "Calculator": CalculatorWorker(),
                   "LLM": LLMWorker(),
                   "SearchSOTU": SearchSOTUWorker()}
