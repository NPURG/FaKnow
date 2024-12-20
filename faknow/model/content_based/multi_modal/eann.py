from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from faknow.model.layers.layer import GradientReverseLayer, TextCNNLayer
from faknow.model.model import AbstractModel


class EANN(AbstractModel):
    r"""
    EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection, KDD 2018
    paper: https://dl.acm.org/doi/abs/10.1145/3219819.3219903
    code: https://github.com/yaqingwang/EANN-KDD18
    """

    def __init__(self,
                 event_num: int,
                 embed_weight: torch.Tensor,
                 reverse_lambda=1.0,
                 hidden_size=32):
        """
        Args:
            event_num (int): number of events
            embed_weight (Tensor): weights for word embedding layer, shape=(vocab_size, embedding_size)
            reverse_lambda (float): lambda for gradient reverse layer. Default=1
            hidden_size (int): size for hidden layers. Default=32
        """
        super(EANN, self).__init__()

        self.event_num = event_num
        self.embed_dim = embed_weight.shape[-1]
        self.hidden_size = hidden_size
        self.reverse_lambda = reverse_lambda

        # text
        self.embed = nn.Embedding.from_pretrained(embed_weight)

        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.text_cnn_layer = TextCNNLayer(self.embed_dim, filter_num,
                                           window_size, F.leaky_relu)
        self.text_fc = nn.Linear(
            len(window_size) * filter_num, self.hidden_size)

        # image
        vgg_19 = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT)
        for param in vgg_19.parameters():
            param.requires_grad = False

        self.vgg = vgg_19
        self.image_fc = nn.Linear(vgg_19.classifier._modules['6'].out_features, self.hidden_size)

        # Class Classifier
        self.class_classifier = nn.Linear(2 * self.hidden_size, 2)

        # Event Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size, self.event_num))

    def forward(self, token_id: torch.Tensor,
                mask: torch.Tensor, image: torch.Tensor):
        """

        Args:
            token_id (Tensor): text token ids
            image (Tensor): image pixels
            mask (Tensor): text masks

        Returns:
            tuple:
                - class_output (Tensor): prediction of being fake news, shape=(batch_size, 2)
                - domain_output (Tensor): prediction of belonging to which domain, shape=(batch_size, domain_num)
        """
        # IMAGE
        image = self.vgg(image)  # [N, 512]
        image = F.leaky_relu(self.image_fc(image))

        # text CNN
        text = self.embed(token_id)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = self.text_cnn_layer(text)
        text = F.leaky_relu(self.text_fc(text))

        # combine Text and Image
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)
        # Domain (which Event)
        reverse_feature = GradientReverseLayer.apply(text_image,
                                                     self.reverse_lambda)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def calculate_loss(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        calculate total loss, classification loss and domain loss via CrossEntropyLoss,
        where total loss = classification loss + domain loss

        Args:
            data (Dict[str, Any]): batch data dict

        Returns:
            Dict[str, Tensor]: loss dict, key: total_loss, class_loss, domain_loss
        """
        token_id = data['text']['token_id']
        mask = data['text']['mask']
        image = data['image']
        event_label = data['domain']
        label = data['label']
        class_output, domain_output = self.forward(token_id, mask, image)

        criterion = nn.CrossEntropyLoss()
        class_loss = criterion(class_output, label)
        domain_loss = criterion(domain_output, event_label)
        loss = class_loss + domain_loss

        return {'total_loss': loss, 'class_loss': class_loss, 'domain_loss': domain_loss}

    def predict(self, data_without_label) -> torch.Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """
        token_id = data_without_label['text']['token_id']
        mask = data_without_label['text']['mask']
        image = data_without_label['image']
        class_output, _ = self.forward(token_id, mask, image)
        return torch.softmax(class_output, dim=-1)
