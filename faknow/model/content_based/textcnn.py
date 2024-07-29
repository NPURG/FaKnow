from typing import Callable, Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from faknow.model.layers.layer import TextCNNLayer
from faknow.model.model import AbstractModel


class TextCNN(AbstractModel):
    r"""
    Convolutional Neural Networks for Sentence Classification, EMNLP 2014
    paper: https://aclanthology.org/D14-1181/
    code: https://github.com/yoonkim/CNN_sentence
    """

    def __init__(self,
                 word_vectors: torch.Tensor,
                 filter_num=100,
                 kernel_sizes: List[int] = None,
                 activate_func: Optional[Callable] = F.relu,
                 dropout=0.5,
                 freeze=False):
        """
        Args:
            word_vectors (torch.Tensor): weights of word embedding layer, shape=(vocab_size, embedding_size)
            filter_num (int): number of filters in conv layer. Default=100
            kernel_sizes (List[int]): list of different kernel_num sizes for TextCNNLayer. Default=[3, 4, 5]
            activate_func (Callable): activate function for TextCNNLayer. Default=relu
            dropout (float): drop out rate of fully connected layer. Default=0.5
            freeze (bool): whether to freeze weights in word embedding layer while training. Default=False
        """

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
        """
        Args:
            text: batch data, shape=(batch_size, max_len)

        Returns:
            Tensor: output, shape=(batch_size, 2)
        """

        text = self.word_embedding(text)
        text = self.text_ccn_layer(text)
        out = self.classifier(text)
        return out

    def calculate_loss(self, data) -> Tensor:
        """
        calculate loss via CrossEntropyLoss

        Args:
            data: batch data tuple

        Returns:
            torch.Tensor: loss
        """

        text = data['text']
        label = data['label']
        out = self.forward(text)
        loss = self.loss_func(out, label)
        return loss

    def predict(self, data_without_label):
        """
        predict the probability of being fake news

        Args:
            data_without_label: batch data

        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """

        text = data_without_label['text']
        out = self.forward(text)
        return F.softmax(out, dim=-1)
