# Planner is a LLMNode to reason without prior. It can call for tool usage (Workers)
from nodes.LLMNode import LLMNode
from nodes.Worker import WORKER_REGISTRY
from prompts.planner import *


# DEFAULT_PREFIX = "For the following tasks, make plans that can solve the problem step-by-step. For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a variable #E (#E1, #E2, ...) that can be called by later tools. Available tools are: \n\n"
# DEFAULT_SUFFIX = "Begin! Describe your plans with rich details.\n\n"
# DEFAULT_FEWSHOT = "\n"


class Planner(LLMNode):
    def __init__(self, workers, prefix=DEFAULT_PREFIX, suffix=DEFAULT_SUFFIX, fewshot=DEFAULT_FEWSHOT,
                 model_name="text-davinci-003", stop=None):
        super().__init__("Planner", model_name, stop, input_type=str, output_type=str)
        self.workers = workers
        self.prefix = prefix
        self.worker_prompt = self._generate_worker_prompt()
        self.suffix = suffix
        self.fewshot = fewshot

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        prompt = self.prefix + self.worker_prompt + self.suffix + self.fewshot + input + '\n'
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def _get_worker(self, name):
        if name in WORKER_REGISTRY:
            return WORKER_REGISTRY[name]
        else:
            raise ValueError("Worker not found")

    def _generate_worker_prompt(self):
        prompt = "Tools can be one of the following:\n"
        for name in self.workers:
            worker = self._get_worker(name)
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt + "\n"
