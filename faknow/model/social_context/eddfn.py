from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from faknow.model.layers.transformer import FFN
from faknow.model.model import AbstractModel


class _Discriminator(nn.Module):
    """
    Discriminator in EDDFN
    """
    def __init__(self, input_size: int, domain_size: int):
        super().__init__()
        self.ffn = FFN(input_size,
                       domain_size * 2,
                       domain_size,
                       activation=torch.sigmoid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_representation: Tensor):
        return self.sigmoid(self.ffn(input_representation))


class EDDFN(AbstractModel):
    r"""
    Embracing Domain Differences in Fake News Cross-domain Fake News Detection using Multi-modal Data, AAAI 2021
    paper: https://ojs.aaai.org/index.php/AAAI/article/view/16134
    code: https://github.com/amilasilva92/cross-domain-fake-news-detection-aaai2021
    """
    def __init__(self,
                 input_size: int,
                 domain_size: int,
                 lambda1=1.0,
                 lambda2=10.0,
                 lambda3=5.0,
                 hidden_size=512):
        """
        Args:
            input_size (int): dimension of input representation
            domain_size (int): dimension of domain vector
            lambda1 (float): L_{recon} loss weight. Default=1.0
            lambda2 (float): L_{specific} loss weight. Default=10.0
            lambda3 (float): L_{shared} loss weight. Default=5.0
            hidden_size (int): size of hidden layer. Default=512
        """

        super().__init__()
        self.input_size = input_size
        self.domain_size = domain_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.discriminate_size = int(hidden_size / 2)

        self.input_embedding_layer = nn.Sequential(
            FFN(input_size, self.discriminate_size, hidden_size), nn.ReLU())

        self.output_classifier = nn.Sequential(nn.Linear(hidden_size, 1),
                                               nn.Sigmoid())
        self.output_decoder = FFN(hidden_size, self.discriminate_size,
                                  input_size)

        self.specific_discriminator = _Discriminator(self.discriminate_size,
                                                     domain_size)
        self.shared_discriminator = _Discriminator(self.discriminate_size,
                                                   domain_size)

    def forward(
            self, input_representation: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            input_representation (Tensor): input representation consisting of text and propagation network

        Returns:
            tuple:
                class_out (Tensor): prediction of being fake news, shape=(batch_size, 1)
                decoder_out (Tensor): prediction of input, shape=(batch_size, input_size)
                specific_domain (Tensor): specific domain output, shape=(batch_size, domain_size)
                shared_domain (Tensor): shared domain output, shape=(batch_size, domain_size)
        """

        input_embedding = self.input_embedding_layer(input_representation)

        # shape=(batch_size * 1)
        class_out = self.output_classifier(input_embedding)

        decoder_out = self.output_decoder(input_embedding)

        specific_input = input_embedding[:, :self.discriminate_size]
        shared_input = input_embedding[:, self.discriminate_size:]

        specific_domain = self.specific_discriminator(specific_input)
        shared_domain = self.shared_discriminator(shared_input)

        return class_out, decoder_out, specific_domain, shared_domain

    def calculate_loss(self, data: Tuple[Tensor]) -> Dict[str, Tensor]:
        """
        calculate total loss,
        including classification loss(BCELoss), reconstruction loss(MSELoss),
        specific domain loss(MSELoss) and shared domain loss(MSELoss)

        Args:
            data (Tuple[Tensor]): batch data tuple, including input, domain and label

        Returns:
            Dict[str, Tensor]: loss dict, key: total_loss, class_loss, decoder_loss, specific_domain_loss, shared_domain_loss
        """

        input, domain, label = data
        class_out, decoder_out, specific_domain, shared_domain = self.forward(
            input)

        class_loss = nn.BCELoss()(class_out.squeeze(), label.float())
        decoder_loss = nn.MSELoss()(decoder_out, input) * self.lambda1
        specific_domain_loss = nn.MSELoss()(specific_domain,
                                            domain) * self.lambda2
        shared_domain_loss = nn.MSELoss()(shared_domain, domain) * self.lambda3

        loss = class_loss + decoder_loss + specific_domain_loss + shared_domain_loss

        return {
            'total_loss': loss,
            'class_loss': class_loss,
            'decoder_loss': decoder_loss,
            'specific_domain_loss': specific_domain_loss,
            'shared_domain_loss': shared_domain_loss
        }

    def predict(self, data_without_label: Tuple[Tensor]):
        """
        predict the probability of being fake news

        Args:
            data_without_label (Tuple[Tensor]): batch data tuple, including input, domain

        Returns:
            Tensor: one hot probability, shape=(batch_size, 2)
        """

        if type(data_without_label) is tuple:
            input_representation = data_without_label[0]
        else:
            input_representation = data_without_label

        class_out = self.forward(input_representation)[0]
        class_out = torch.round(class_out).long().squeeze()
        return F.one_hot(class_out, num_classes=2)
