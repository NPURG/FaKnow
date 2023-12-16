import datetime
import warnings
from typing import Dict

import torch


def dict2str(result_dict: Dict[str, float]) -> str:
    """
    Convert a dictionary of metrics to a string.

    Args:
        result_dict (dict): A dictionary containing metric names and corresponding values.

    Returns:
        str: The formatted string representation of the dictionary.
    """

    return "    ".join([
        str(metric) + "=" + f"{value:.6f}"
        for metric, value in result_dict.items()
    ])


def now2str() -> str:
    """
    Get the current time and convert it to a formatted string.

    Returns:
        str: The current time in the format '%Y-%m-%d-%H_%M_%S'.
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%Y-%m-%d-%H_%M_%S")

    return cur


def seconds2str(seconds: float) -> str:
    """
    Convert seconds to a human-readable time format.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The duration in the format 'h:mm:ss' or 'm:ss'.
    """
    if seconds < 60:
        return f'{seconds:.6f}s'

    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if m == 0:
        return f'{s}s'
    elif h == 0:
        return f'{m}m{s}s'
    return f'{h}h{m}m{s}s'


def check_loss_type(result):
    """
    Check the type of the loss and convert it to a tensor if necessary.

    Args:
        result (Union[torch.Tensor, dict]): The loss value or a dictionary of losses.

    Returns:
        Tuple[torch.Tensor, bool]: A tuple containing the loss tensor and a boolean indicating if the loss was a dictionary.
    """
    result_is_dict = False

    if type(result) is dict:
        result_is_dict = True
        if 'total_loss' in result.keys():
            loss = result['total_loss']
        else:
            # todo 是否允许没有total_loss，采用所有loss的和作为total_loss
            warnings.warn(
                f"no total_loss in result: {result}, use sum of all losses as total_loss"
            )
            loss = torch.sum(torch.stack(list(result.values())))
    elif type(result) is torch.Tensor:
        loss = result
    else:
        raise TypeError(f"result type error: {result}")

    return loss, result_is_dict


class EarlyStopping:
    """
    Early stopping to stop the training when the score does not improve after
    certain epochs.
    """

    def __init__(self, patience=10, delta=0.000001, mode='max'):
        """
        Args:
            patience (int): number of epochs to wait for improvement, default=10
            delta (float): minimum change in the monitored quantity to qualify as an improvement, default=0.000001
            mode (str): minimize or maximize score, one of {min, max}, default=max
        """

        assert mode in ['min', 'max'], "mode must be either 'min' or 'max'"

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score: float) -> bool:
        """
        Check if the current score is the best score and update early stopping status.

        Args:
            current_score (float): The current score to check if it is the best score.

        Returns:
            bool: Whether the current score is the best score.
        """
        improvement = False
        if self.best_score is None:
            improvement = True
        elif self.mode == 'min':
            if current_score < self.best_score - self.delta:
                improvement = True
        elif self.mode == 'max':
            if current_score > self.best_score + self.delta:
                improvement = True

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return improvement
