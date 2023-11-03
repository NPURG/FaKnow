import torch
import torch.nn as nn
from torch.nn import GRU
from torch.nn import Embedding
from torch import Tensor
from faknow.model.model import AbstractModel

class DUDEF(AbstractModel):
    r"""
    DUDEF: Mining Dual Emotion for Fake News Detection, WWW 2021
    paper: https://arxiv.org/abs/1903.01728
    code: https://github.com/RMSnow/WWW2021
    """
    def __init__(self,
                 input_size,
                 emotion_len,
                 hidden_size,
                 embedding_matrix):

        """
        Args:
            input_size (int): dimension of input node feature
            emotion_len(int): dimension of dual_emotion
            hidden_size (int): dimension of hidden layer
            embedding_matrix(ndarray): embeddings initializer matrix
        """

        super(DUDEF, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.bi_gru = GRU(input_size=input_size, hidden_size=32, num_layers=2, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size * 2 + emotion_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.global_pooling = nn.AdaptiveAvgPool2d((1,64))
        self.embedding =Embedding.from_pretrained(self.embedding_matrix)
    def forward(self, data) -> Tensor:
        """
        Args:
            dict:
                emotions(Tensor): node emotions features, shape=(batch_size, feature_size),
                senmantics(Tensor):node senmantics features, shape=(batch_size, feature_size)

        Returns:
            Tensor: prediction of being fake, shape=(batch_size, 2)
        """

        emd = self.embedding(data['senmantics'])
        output, _ = self.bi_gru(emd.float())
        output = self.global_pooling(output).squeeze(1)
        output = torch.cat((output,data['emotions'].float()),1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output

    def calculate_loss(self, data):
        """
        calculate loss via CrossEntropyLoss

        Args:
            data (Batch): batch data

        Returns:
            torch.Tensor: loss
        """
        output = self.forward(data['data'])
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(output, data['label'])
        return loss

    def predict(self, data) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data (Batch): batch data

        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """

        output = self.forward(data['data'])
        return output