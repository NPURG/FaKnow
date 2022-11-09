from typing import List
import torch


# accuracy, precision, recall, f1-score, auc ...
class Evaluator:
    def __init__(self, metrics: List):
        self.metrics = metrics

    # def evaluate(self, data):
    #     result = {}
    #     right = 0
    #     total = 0
    #     for outputs, y in data:
    #         right += torch.sum(outputs == y)
    #         total += y.shape[0]
    #     result['accuracy'] = (right / total).item()
    #     return result

    def evaluate(self, outputs: torch.Tensor, Y: torch.Tensor):
        """each output is a vector and each y in Y is a scalar"""
        result = {}
        result['accuracy'] = (torch.sum(outputs.argmax(dim=1) == Y) / Y.shape[0]).item()
        return result
