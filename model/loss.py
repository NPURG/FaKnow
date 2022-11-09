import torch.nn as nn
from  torch.nn.modules.loss import _Loss


class CrossEntropyLoss(_Loss):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self):
        loss = nn.CrossEntropyLoss(reduction='none')
        # loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
