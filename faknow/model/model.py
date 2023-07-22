from typing import Union, Dict

import torch.nn as nn
from torch import Tensor


class AbstractModel(nn.Module):
    """
    abstract class for all models, every model should inherit it and implement the following methods:
    1. calculate_loss
    2. predict
    """

    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, data) -> Union[Tensor, Dict[str, Tensor]]:
        """
        calculate loss

        Args:
            data: batch data

        Returns:
            Union[Tensor, Dict[str, Tensor]]: loss or a dict of loss if there are multiple losses
        """
        raise NotImplementedError

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label: batch data

        Returns:
            Tensor: probability, shape=(batch_size, 2)
        """
        raise NotImplementedError

    # todo 如果用val loss进行early stopping
    # 那么需要 一个函数调用forward，既能计算loss，又能predict
    # 这样进行validate时，能看到loss与accuracy
