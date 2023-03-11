from typing import Callable, Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.layer import TextCNNLayer
from model.model import AbstractModel

"""
Convolutional Neural Networks for Sentence Classification
paper: https://arxiv.org/abs/1408.5882
code: https://github.com/yoonkim/CNN_sentence
"""


class TextCNN(AbstractModel):
    def __init__(self,
                 word_vectors: torch.Tensor,
                 filter_num: int,
                 kernel_sizes: List[int],
                 activate_func: Optional[Callable] = F.relu,
                 dropout=0.5,
                 freeze=True):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=freeze)

        self.text_ccn_layer = TextCNNLayer(word_vectors.shape[-1], filter_num,
                                           kernel_sizes, activate_func)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len(kernel_sizes) * filter_num, 2)
        )

    def forward(self, text: torch.Tensor):
        text = self.word_embedding(text)
        text = self.text_ccn_layer(text)
        out = self.classifier(text)
        return out

    def calculate_loss(self, data):
        text, label = data
        out = self.forward(text)
        loss = self.loss_func(out, label)
        return loss

    def predict(self, data_without_label):
        if type(data_without_label) is tuple:
            text = data_without_label[0]
        else:
            text = data_without_label
        out = self.forward(text)
        return F.softmax(out, dim=-1)
