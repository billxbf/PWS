import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import tqdm
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI

from algos.PWS import PWS_Base, PWS_Extra
from algos.notool import CoT, IO
from algos.react import ReactBase


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# TODO: SUPPORT TRIVIA
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def llm_accuracy_score(query, prediction, ground_truth):
    data = [{
        'query': query,
        'answer': ground_truth,
    }]
    pred = [{
        'query': query,
        'answer': ground_truth,
        'result': prediction,
    }]
    eval_chain = QAEvalChain.from_llm(OpenAI(temperature=0))
    graded_outputs = eval_chain.evaluate(data, pred)
    return 1 if graded_outputs[0]['text'].strip() == 'CORRECT' else 0


class Evaluator:
    def __init__(self, task, dataset, algo, maxtry=1):
        assert task in ["hotpot_qa", "fever", "trivia_qa"]
        assert isinstance(dataset, pd.DataFrame)
        assert isinstance(algo, (PWS_Base, PWS_Extra, ReactBase, IO, CoT))

        self.task = task
        self.dataset = dataset
        self.algo = algo
        self.maxtry = maxtry
        self.failed_response = self._failed_response()

    def run(self):
        print("\n******************* Start Evaluation *******************\n")
        result = {}
        data = {}
        if self.task == "hotpot_qa":
            em, f1, acc, wall_time, steps, total_tokens, token_cost, tool_cost, total_cost, preds = [], [], [], [], [], [], [], [], [], []
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["question"][i]
                label = self.dataset["answer"][i]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                pred = self._parse_prediction(response["output"])
                em += [self.get_metrics(question, label, pred)["em"]]
                f1 += [self.get_metrics(question, label, pred)["f1"]]
                acc += [self.get_metrics(question, label, pred)["acc"]]
                wall_time += [response["wall_time"]]
                total_tokens += [response["total_tokens"]]
                total_cost += [response["total_cost"]]
                steps += [response["steps"]]
                token_cost += [response["token_cost"]]
                tool_cost += [response["tool_cost"]]
                preds += [pred]
            result["avg_em"] = np.nanmean(em)
            result["avg_f1"] = np.nanmean(f1)
            result["avg_acc"] = np.nanmean(acc)
            result["avg_acc"] = np.nanmean(acc)
            result["avg_wall_time"] = np.nanmean(wall_time)
            result["avg_total_tokens"] = np.nanmean(total_tokens)
            result["avg_total_cost"] = np.nanmean(total_cost)
            result["avg_steps"] = np.nanmean(steps)
            result["avg_token_cost"] = np.nanmean(token_cost)
            result["avg_tool_cost"] = np.nanmean(tool_cost)
            data["preds"] = preds
            data["em"] = em
            data["f1"] = f1
            data["acc"] = acc
            data["wall_time"] = wall_time
            data["total_tokens"] = total_tokens
            data["total_cost"] = total_cost
            data["steps"] = steps
            data["token_cost"] = token_cost
            data["tool_cost"] = tool_cost
        elif self.task == "fever":
            em, f1, wall_time, steps, total_tokens, token_cost, tool_cost, total_cost, preds = [], [], [], [], [], [], [], [], []
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["claim"][i]
                label = self.dataset["label"][i]
                for _ in range(self.maxtry):
                    try:
                        response = self.algo.run(question)
                        break
                    except:
                        response = self.failed_response
                pred = self._parse_prediction(response["output"])
                em += [self.get_metrics(label, pred)["em"]]
                f1 += [self.get_metrics(label, pred)["f1"]]
                wall_time += [response["wall_time"]]
                total_tokens += [response["total_tokens"]]
                total_cost += [response["total_cost"]]
                steps += [response["steps"]]
                token_cost += [response["token_cost"]]
                tool_cost += [response["tool_cost"]]
                preds += [pred]
            result["avg_em"] = np.nanmean(em)
            result["avg_f1"] = np.nanmean(f1)
            result["avg_wall_time"] = np.nanmean(wall_time)
            result["avg_total_tokens"] = np.nanmean(total_tokens)
            result["avg_total_cost"] = np.nanmean(total_cost)
            result["avg_steps"] = np.nanmean(steps)
            result["avg_token_cost"] = np.nanmean(token_cost)
            result["avg_tool_cost"] = np.nanmean(tool_cost)
            data["preds"] = preds
            data["em"] = em
            data["f1"] = f1
            data["wall_time"] = wall_time
            data["total_tokens"] = total_tokens
            data["total_cost"] = total_cost
            data["steps"] = steps
            data["token_cost"] = token_cost
            data["tool_cost"] = tool_cost
        elif self.task == "trivia_qa":
            raise NotImplementedError
        return result, data

    def get_metrics(self, query, label, pred):
        if pred is None:
            return {'em': 0, 'f1': 0}
        norm_label = normalize_answer(label)
        norm_pred = normalize_answer(pred)
        em = (norm_pred == norm_label)
        f1 = f1_score(norm_pred, norm_label)
        acc = llm_accuracy_score(query, pred, label)
        return {'em': em, 'f1': f1, 'acc': acc}

    def _parse_prediction(self, output):
        if isinstance(self.algo, IO):
            return str(output).strip("\n")
        elif isinstance(self.algo, CoT):
            return str(output).split("\n")[-1].replace("Answer:", "")
        elif isinstance(self.algo, ReactBase):
            return str(output).strip("\n")
        elif isinstance(self.algo, PWS_Base):
            return str(output).strip("\n")
        elif isinstance(self.algo, PWS_Extra):
            return str(output).strip("\n")

    def _failed_response(self):
        resposne = {}
        for key in ["input", "output", "wall_time", "total_tokens", "total_cost", "steps", "token_cost", "tool_cost"]:
            resposne[key] = np.nan
        return resposne
