import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable


def get_metric_func(name: str) -> Callable:
    name = name.lower()
    if name == 'accuracy':
        return calculate_accuracy
    if name == 'precision':
        return calculate_precision
    if name == 'recall':
        return calculate_recall
    if name == 'f1':
        return calculate_f1
    raise RuntimeError(f'no metric function called {name}')


def calculate_accuracy(outputs: torch.Tensor, y: torch.Tensor):
    return (outputs.argmax(dim=1).detach().cpu() == y.cpu()).float().mean().item()


def calculate_precision(outputs: torch.Tensor, y: torch.Tensor):
    return precision_score(outputs.argmax(dim=1).detach().cpu().numpy(), y.cpu().numpy(), zero_division=0)


def calculate_recall(outputs: torch.Tensor, y: torch.Tensor):
    return recall_score(outputs.argmax(dim=1).detach().cpu().numpy(), y.cpu().numpy(), zero_division=0)


def calculate_f1(outputs: torch.Tensor, y: torch.Tensor):
    return f1_score(outputs.argmax(dim=1).detach().cpu().numpy(), y.cpu().numpy(), zero_division=0)
