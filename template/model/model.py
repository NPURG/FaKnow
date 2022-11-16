import torch.nn as nn


class AbstractModel(nn.Module):
    """abstract class for all models"""
    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
