from typing import Dict
import torch
from template.evaluate.metrics import get_metric_func


class Evaluator:
    def __init__(self, metrics: Dict):
        """generate metric functions for evaluator
        Args:
            metrics: the key should be metric name:str, value should be self defined metric function.
            if value is None, built-in metric functions will be called according to the metric name
        """
        self.metrics = {
            metric_name: get_metric_func(metric_name)
            if metric_func is None else metric_func
            for metric_name, metric_func in metrics.items()
        }

    # def evaluate(self, data):
    #     result = {}
    #     right = 0
    #     total = 0
    #     for outputs, y in data:
    #         right += torch.sum(outputs == y)
    #         total += y.shape[0]
    #     result['accuracy'] = (right / total).item()
    #     return result

    def evaluate(self, outputs: torch.Tensor, y: torch.Tensor):
        """each output in outputs is a vector and y should be a vector"""
        # result = {}
        # result['accuracy'] = calculate_accuracy(outputs, y)
        # result['precision'] = calculate_precision(outputs, y, 'micro')
        result = {
            metric_name: metric_func(outputs, y)
            for metric_name, metric_func in self.metrics.items()
        }
        return result
