from typing import Dict, Callable, List, Union

import torch
from torch import Tensor

from faknow.evaluate.metrics import get_metric_func


class Evaluator:
    def __init__(self, metrics: List[Union[str, Callable[[Tensor, Tensor], float]]] = None):
        """Initialize the Evaluator.

        Args:
            metrics (List[Union[str, Callable[[Tensor, Tensor], float]]], optional):
                A list of metrics, either as strings or Callable functions.
                If the metric is a string, built-in metric functions (`accuracy`, `precision`, `recall`, `f1`)
                will be used based on the metric name.
                If the metric is a Callable function with signature `metric_func(outputs: Tensor, y: Tensor) -> float`,
                it will be used directly as the metric function.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.metrics = {}

        for metric in metrics:
            if type(metric) == str:
                # If the metric is a string, fetch the corresponding built-in metric function.
                self.metrics[metric] = get_metric_func(metric)
            elif isinstance(metric, Callable):
                 # If the metric is a Callable, use it as is and add it to the metrics dictionary.
                self.metrics[metric.__name__] = metric
            else:
                raise RuntimeError(f'only str or callable are supported as metrics, but {type(metric)} are provided')

    def evaluate(self, outputs: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model's performance using the provided metrics.

        Args:
            outputs (torch.Tensor): Model's predictions.
            y (torch.Tensor): Ground truth labels.

        Returns:
            Dict[str, float]: A dictionary containing metric names as keys and their corresponding values as floats.
        """
        result = {
            metric_name: metric_func(outputs, y)
            for metric_name, metric_func in self.metrics.items()
        }
        return result
