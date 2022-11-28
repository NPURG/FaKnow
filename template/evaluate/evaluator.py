from typing import Dict, Callable, Optional, List, Union
import torch
from template.evaluate.metrics import get_metric_func


class Evaluator:
    def __init__(self, metrics: List[Union[str, Callable]]):
        """generate metric functions for evaluator
        Args:
            metrics: the key should be metric name:str, value should be self defined metric function.
            if value is None, built-in metric functions will be called according to the metric name
        """
        # self.metrics = {
        #     metric: get_metric_func(metric)
        #     if type(metric) == str else metric_func
        #     for metric in metrics
        # }
        self.metrics = {}
        for metric in metrics:
            if type(metric) == str:
                self.metrics[metric] = get_metric_func(metric)
            elif type(metric) == Callable:
                self.metrics[metric.__name__] = metric
            else:
                raise RuntimeError(f'only str or callable are supported as metrics, but {type(metric)} are provided')

    # def evaluate(self, data):
    #     result = {}
    #     right = 0
    #     total = 0
    #     for outputs, y in data:
    #         right += torch.sum(outputs == y)
    #         total += y.shape[0]
    #     result['accuracy'] = (right / total).item()
    #     return result

    def evaluate(self, outputs: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """each output in outputs is a vector and y should be a vector"""
        result = {
            metric_name: metric_func(outputs, y)
            for metric_name, metric_func in self.metrics.items()
        }
        return result
