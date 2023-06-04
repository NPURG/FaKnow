from typing import Union, Dict

import torch.nn as nn
from torch import Tensor


class AbstractModel(nn.Module):
    """abstract class for all models"""

    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, data) -> Union[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError

    def predict(self, data_without_label) -> Tensor:
        raise NotImplementedError

    # todo 如果用val loss进行early stopping
    # 那么需要 一个函数调用forward，既能计算loss，又能predict
    # 这样进行validate时，能看到loss与accuracy
