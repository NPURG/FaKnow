from typing import Callable, Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from faknow.model.layers.layer import TextCNNLayer
from faknow.model.model import AbstractModel

"""
Convolutional Neural Networks for Sentence Classification
paper: https://aclanthology.org/D14-1181/
code: https://github.com/yoonkim/CNN_sentence
"""


class TextCNN(AbstractModel):
    r"""TextCNN model with a TextCNN layer and a fully connected layer

        Args:
            word_vectors (torch.Tensor): weights of word embedding layer, shape=(vocab_size, embedding_size)
            filter_num (int): number of filters in conv layer. Default=100
            kernel_sizes (List[int]): list of different kernel_num sizes for TextCNNLayer. Default=[3, 4, 5]
            activate_func (Callable): activate function for TextCNNLayer. Default=relu
            dropout (float): drop out rate of fully connected layer. Default=0.5
            freeze (bool): whether to freeze weights in word embedding layer while training. Default=False
        """

    def __init__(self,
                 word_vectors: torch.Tensor,
                 filter_num=100,
                 kernel_sizes: List[int] = None,
                 activate_func: Optional[Callable] = F.relu,
                 dropout=0.5,
                 freeze=False):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.loss_func = nn.CrossEntropyLoss()

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=freeze)

        self.text_ccn_layer = TextCNNLayer(word_vectors.shape[-1], filter_num,
                                           kernel_sizes, activate_func)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len(kernel_sizes) * filter_num, 2)
        )

    def forward(self, text: torch.Tensor) -> Tensor:
        text = self.word_embedding(text)
        text = self.text_ccn_layer(text)
        out = self.classifier(text)
        return out

    def calculate_loss(self, data) -> Tensor:
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
