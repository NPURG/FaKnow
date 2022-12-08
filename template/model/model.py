from typing import Optional, Callable, List

import torch.nn as nn


class AbstractModel(nn.Module):
    """abstract class for all models"""

    def __init__(self, loss_funcs: Optional[List[Callable]] = None, loss_weights: Optional[List[float]] = None):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, data):
        raise NotImplementedError

    def predict(self, data_without_label):
        raise NotImplementedError
