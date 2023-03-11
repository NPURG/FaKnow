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


class EDDF(AbstractModel):

    def __init__(self, input_size: int, domain_size: int, lambda1: float, lambda2: float, lambda3: float,
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

    def forward(self, input: Tensor):
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
        decoder_loss = nn.MSELoss()(decoder_out, input) * self.lambda1
        specific_domain_loss = nn.MSELoss()(specific_domain, domain) * self.lambda2
        shared_domain_loss = nn.MSELoss()(shared_domain, domain) * self.lambda3

        loss = class_loss + decoder_loss + specific_domain_loss - shared_domain_loss

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