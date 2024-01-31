from typing import Dict

import torch
import torch.nn as nn
from torch.nn import GRU
from torch.nn import Embedding
from torch import Tensor

from faknow.model.model import AbstractModel


class DUDEF(AbstractModel):
    r"""
    DUDEF: Mining Dual Emotion for Fake News Detection, WWW 2021
    paper: https://dl.acm.org/doi/10.1145/3442381.3450004
    code: https://github.com/RMSnow/WWW2021
    """
    def __init__(self, input_size: int, emotion_len: int, hidden_size: int,
                 embedding_matrix: Tensor):
        """
        Args:
            input_size (int): dimension of input node feature
            emotion_len(int): dimension of dual_emotion
            hidden_size (int): dimension of hidden layer
            embedding_matrix(Tensor): word embedding matrix
        """

        super(DUDEF, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.bi_gru = GRU(input_size=input_size,
                          hidden_size=32,
                          num_layers=2,
                          bidirectional=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size * 2 + emotion_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 64))
        self.embedding = Embedding.from_pretrained(self.embedding_matrix)

    def forward(self, semantics: Tensor, emotions: Tensor) -> Tensor:
        """
        Args:
            semantics(Tensor):node semantics features, shape=(batch_size, feature_size)
            emotions(Tensor): node emotions features, shape=(batch_size, feature_size)

        Returns:
            Tensor: prediction of being fake, shape=(batch_size, 2)
        """

        emd = self.embedding(semantics)
        output, _ = self.bi_gru(emd.float())
        output = self.global_pooling(output).squeeze(1)
        output = torch.cat((output, emotions), 1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output

    def calculate_loss(self, data: Dict[str, Tensor]) -> Tensor:
        """
        calculate loss via CrossEntropyLoss

        Args:
            data (Dict[str, Tensor]): batch data, shape=(2,batch_size,feature_size)

        Returns:
            Tensor: loss
        """
        output = self.forward(data['semantics'], data['emotions'])
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(output, data['label'])
        return loss

    def predict(self, data: Dict[str, Tensor]) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data (Batch): batch data, shape=(2,batch_size,feature_size)

        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """

        output = self.forward(data['semantics'], data['emotions'])
        return output
