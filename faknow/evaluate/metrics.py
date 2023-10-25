import torch
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score)
from typing import Callable


def get_metric_func(name: str) -> Callable:
    """Get the appropriate metric function based on the given name.

    Args:
        name (str): The name of the metric function.

    Returns:
        Callable: The corresponding metric function.

    Raises:
        RuntimeError: If no metric function with the provided name is found.
    """
    name = name.lower()
    if name == 'accuracy':
        return calculate_accuracy
    if name == 'precision':
        return calculate_precision
    if name == 'recall':
        return calculate_recall
    if name == 'f1':
        return calculate_f1
    if name == 'auc':
        return calculate_auc
    raise RuntimeError(f'no metric function called {name}')


def calculate_accuracy(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the accuracy metric.

    Args:
        outputs (torch.Tensor): Model's predictions.
        y (torch.Tensor): Ground truth labels.

    Returns:
        float: The accuracy value.
    """
    return (outputs.argmax(dim=1).detach().cpu() == y.cpu()).float().mean().item()


def calculate_precision(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the precision metric.

    Args:
        outputs (torch.Tensor): Model's predictions.
        y (torch.Tensor): Ground truth labels.

    Returns:
        float: The precision value.
    """
    return precision_score(y.cpu().numpy(),
                           outputs.argmax(dim=1).detach().cpu().numpy(),
                           zero_division=0)


def calculate_recall(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the recall metric.

    Args:
        outputs (torch.Tensor): Model's predictions.
        y (torch.Tensor): Ground truth labels.

    Returns:
        float: The recall value.
    """
    return recall_score(y.cpu().numpy(),
                        outputs.argmax(dim=1).detach().cpu().numpy(),
                        zero_division=0)


def calculate_f1(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the F1 score metric.

    Args:
        outputs (torch.Tensor): Model's predictions.
        y (torch.Tensor): Ground truth labels.

    Returns:
        float: The F1 score value.
    """
    return f1_score(y.cpu().numpy(),
                    outputs.argmax(dim=1).detach().cpu().numpy(),
                    zero_division=0)


def calculate_auc(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate the AUC score metric.

    Args:
        outputs (torch.Tensor): Model's predictions.
        y (torch.Tensor): Ground truth labels.

    Returns:
        float: The AUC score value.
    """

    return roc_auc_score(y.cpu().numpy(),
                         outputs.argmax(dim=1).detach().cpu().numpy())
