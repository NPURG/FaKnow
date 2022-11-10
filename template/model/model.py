import torch.nn as nn


class Model(nn.Module):
    """abstract class for all models"""
    def __init__(self):
        super(Model, self).__init__()

    def calculate_loss(self):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
