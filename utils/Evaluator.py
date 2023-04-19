import datasets
from utils.DataLoader import DataLoader
from algos.PWS import PWS_Base, PWS_Unlimited
from algos.react import ReactBase
import pandas as pd
import re
import string
from collections import Counter
import tqdm
import numpy as np
from prompts.fewshots import *

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

class Evaluator:
    def __init__(self, task, dataset, algo):
        assert task in ["hotpot_qa", "fever", "trivia_qa"]
        assert isinstance(dataset, pd.DataFrame)
        assert isinstance(algo, (PWS_Base, PWS_Unlimited, ReactBase))

        self.task = task
        self.dataset = dataset
        self.algo = algo

    def run(self):
        print("\n******************* Start Evaluation *******************\n")
        result = {}
        if self.task == "hotpot_qa":
            em, f1, wall_time, total_tokens, total_cost = [], [], [], [], []
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["question"][i]
                label = self.dataset["answer"][i]
                response = self.algo.run(question)
                pred = response["output"]
                em += [self.get_metrics(label, pred)["em"]]
                f1 += [self.get_metrics(label, pred)["f1"]]
                wall_time += [response["wall_time"]]
                total_tokens += [response["total_tokens"]]
                total_cost += [response["total_cost"]]
            result["avg_em"] = np.mean(em)
            result["avg_f1"] = np.mean(f1)
            result["avg_wall_time"] = np.mean(wall_time)
            result["avg_total_tokens"] = np.mean(total_tokens)
            result["avg_total_cost"] = np.mean(total_cost)
        elif self.task == "fever":
            em, f1, wall_time, total_tokens, total_cost = [], [], [], [], []
            for i in tqdm.tqdm(range(len(self.dataset))):
                question = self.dataset["claim"][i]
                label = self.dataset["label"][i]
                response = self.algo.run(question)
                pred = response["output"]
                em += [self.get_metrics(label, pred)["em"]]
                f1 += [self.get_metrics(label, pred)["f1"]]
                wall_time += [response["wall_time"]]
                total_tokens += [response["total_tokens"]]
                total_cost += [response["total_cost"]]
            result["avg_em"] = np.mean(em)
            result["avg_f1"] = np.mean(f1)
            result["avg_wall_time"] = np.mean(wall_time)
            result["avg_total_tokens"] = np.mean(total_tokens)
            result["avg_total_cost"] = np.mean(total_cost)
        elif self.task == "trivia_qa":
            raise NotImplementedError
        return result

    def get_metrics(self, label, pred):
        if pred is None:
            return {'em': 0, 'f1': 0}
        norm_label = normalize_answer(label)
        norm_pred = normalize_answer(pred)
        em = (norm_pred == norm_label)
        f1 = f1_score(norm_pred, norm_label)
        return {'em': em, 'f1': f1}



