from typing import Dict, Callable, List, Union

import torch

from template.evaluate.metrics import get_metric_func


class Evaluator:
    def __init__(self, metrics: List[Union[str, Callable]]=None):
        """generate metric functions for evaluator
        Args:
            metrics: a list of metrics, str or Callable.
                     If the metric is str, built-in metric functions(`accuracy`, `precision`, `recall`, `f1`) will be called according to the metric name.
                     Or the passing metric function would be called
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.metrics = {}

        for metric in metrics:
            if type(metric) == str:
                self.metrics[metric] = get_metric_func(metric)
            elif isinstance(metric, Callable):
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
        result = {
            metric_name: metric_func(outputs, y)
            for metric_name, metric_func in self.metrics.items()
        }
        return result
