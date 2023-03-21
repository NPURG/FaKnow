from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from model.model import AbstractModel
from template.model.layers.attention import FFN

"""
Embracing Domain Differences in Fake News Cross-domain Fake News Detection using Multi-modal Data
paper: https://arxiv.org/abs/2102.06314
code: github.com/amilasilva92/cross-domain-fake-news-detection-aaai2021
"""


class _Discriminator(nn.Module):
    def __init__(self, input_size: int, domain_size: int):
        super().__init__()
        self.ffn = FFN(input_size, domain_size * 2, domain_size, activation=torch.sigmoid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor):
        return self.sigmoid(self.ffn(input))


class EDDFN(AbstractModel):
    r"""Embracing Domain Differences in Fake News Cross-domain Fake News Detection using Multi-modal Data

        Args:
            input_size (int): dimension of input representation
            domain_size (int): dimension of domain vector
            lambda1 (float): L_{recon} loss weight. Default: 1.0
            lambda2 (float): L_{specific} loss weight. Default: 10.0
            lambda3 (float): L_{shared} loss weight. Default: 5.0
            hidden_size (int): size of hidden layer. Default: 512
    """

    def __init__(self, input_size: int, domain_size: int, lambda1=1.0, lambda2=10.0, lambda3=5.0,
                 hidden_size=512):

        super().__init__()
        self.input_size = input_size
        self.domain_size = domain_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.discriminate_size = int(hidden_size / 2)

        self.input_embedding_layer = nn.Sequential(FFN(input_size, self.discriminate_size, hidden_size), nn.ReLU())

        self.output_classifier = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.output_decoder = FFN(hidden_size, self.discriminate_size, input_size)

        self.specific_discriminator = _Discriminator(self.discriminate_size, domain_size)
        self.shared_discriminator = _Discriminator(self.discriminate_size, domain_size)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        Args:
            input (Tensor): input representation consisting of text and propagation network

        Returns:
            tuple:
                - class_out (Tensor): prediction of being fake news, shape=(batch_size, 1)
                - decoder_out (Tensor): prediction of input, shape=(batch_size, input_size)
                - specific_domain (Tensor): specific domain output, shape=(batch_size, domain_size)
                - shared_domain (Tensor): shared domain output, shape=(batch_size, domain_size)

        """
        input_embedding = self.input_embedding_layer(input)

        # shape=(batch_size * 1)
        class_out = self.output_classifier(input_embedding)

        decoder_out = self.output_decoder(input_embedding)

        specific_input = input_embedding[:, :self.discriminate_size]
        shared_input = input_embedding[:, self.discriminate_size:]

        specific_domain = self.specific_discriminator(specific_input)
        shared_domain = self.shared_discriminator(shared_input)

        return class_out, decoder_out, specific_domain, shared_domain

    def calculate_loss(self, data) -> Tuple[torch.FloatTensor, str]:
        input, domain, label = data
        class_out, decoder_out, specific_domain, shared_domain = self.forward(input)

        class_loss = nn.BCELoss()(class_out.squeeze(), label.float())
        decoder_loss = nn.MSELoss()(decoder_out, input)
        specific_domain_loss = nn.MSELoss()(specific_domain, domain)
        shared_domain_loss = nn.MSELoss()(shared_domain, domain)

        loss = class_loss + decoder_loss * self.lambda1 + specific_domain_loss * self.lambda2 + shared_domain_loss * self.lambda3

        msg = f'class_loss={class_loss}, decoder_loss={decoder_loss}, specific_domain_loss={specific_domain_loss}, ' \
              f'shared_domain_loss={shared_domain_loss} '
        return loss, msg

    def predict(self, data_without_label):
        if type(data_without_label) is tuple:
            input = data_without_label[0]
        else:
            input = data_without_label

        class_out = self.forward(input)[0]
        class_out = torch.round(class_out).long().squeeze()
        return F.one_hot(class_out, num_classes=2)
