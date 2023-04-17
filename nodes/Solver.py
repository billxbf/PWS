# Solver summarizes the plan and evidence of the Planner and Workers.
from nodes.LLMNode import LLMNode
from prompts.solver import *
# DEFAULT_PREFIX = "Solve the following task or problem. To assist you, we provide some plans that can be helpful towards the final solution. You should refer to the evidence for each plan if you find it helpful.\n\n"
# DEFAULT_SUFFIX = "\nNow begin to solve the task or problem. But do not disclose that we've provided you the evidence.\n"

class Solver(LLMNode):
    def __init__(self, prefix=DEFAULT_PREFIX, suffix=DEFAULT_SUFFIX, model_name="text-davinci-003", stop=None):
        super().__init__("Solver", model_name, stop, input_type=str, output_type=str)
        self.prefix = prefix
        self.suffix = suffix

    def run(self, input, worker_log, log=False):
        assert isinstance(input, self.input_type)
        prompt = self.prefix + input + "\n" + worker_log + self.suffix + input + '\n'
        response = self.call_llm(prompt, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion
