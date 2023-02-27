from collections import OrderedDict
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.layers.layer import GradientReverseLayer, TextCNNLayer
from template.model.model import AbstractModel

"""
EANN: Multi-Modal Fake News Detection
paper: https://dl.acm.org/doi/epdf/10.1145/3219819.3219903
code: https://github.com/yaqingwang/EANN-KDD18
"""


class EANN(AbstractModel):
    def __init__(self,
                 event_num: int,
                 hidden_size: int,
                 reverse_lambd: int,
                 embed_weight: np.ndarray,
                 vocab_size: int,
                 loss_funcs: Optional[List[Callable]] = None,
                 loss_weights: Optional[List[float]] = None):
        super(EANN, self).__init__()

        if loss_funcs is None:
            loss_funcs = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
        self.loss_funcs = loss_funcs
        if loss_weights is None:
            loss_weights = [1.0, 1.0]
        self.loss_weights = loss_weights

        self.event_num = event_num
        self.embed_dim = embed_weight[0].shape[0]
        self.hidden_size = hidden_size
        self.reverse_lambd = reverse_lambd

        # 真正的word embedding，使用了预训练好的权重
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embed_weight))

        # TEXT CNN
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.text_ccn_layer = TextCNNLayer(self.embed_dim, filter_num,
                                           window_size, F.leaky_relu)
        self.text_ccn_fc = nn.Linear(
            len(window_size) * filter_num, self.hidden_size)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT)
        for param in vgg_19.parameters():
            param.requires_grad = False

        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc = nn.Linear(num_ftrs, self.hidden_size)

        # Class Classifier
        self.class_classifier = nn.Sequential(
            OrderedDict([('c_fc1', nn.Linear(2 * self.hidden_size, 2)),
                         ('c_softmax', nn.Softmax(dim=1))]))

        # Event Classifier
        self.domain_classifier = nn.Sequential(
            OrderedDict([('d_fc1',
                          nn.Linear(2 * self.hidden_size, self.hidden_size)),
                         ('d_relu1', nn.LeakyReLU(True)),
                         ('d_fc2', nn.Linear(self.hidden_size,
                                             self.event_num)),
                         ('d_softmax', nn.Softmax(dim=1))]))

    def forward(self, text: torch.Tensor, image: torch.Tensor,
                mask: torch.Tensor):
        # IMAGE
        image = self.vgg(image)  # [N, 512]
        image = F.leaky_relu(self.image_fc(image))

        # text CNN
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = self.text_ccn_layer(text)
        text = F.leaky_relu(self.text_ccn_fc(text))

        # combine Text and Image
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)
        # Domain (which Event)
        reverse_feature = GradientReverseLayer.apply(text_image,
                                                     self.reverse_lambd)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def calculate_loss(self, data) -> Tuple[torch.Tensor, str]:
        text, mask, image, event_label, label = data[0][0], data[0][1], data[
            1], data[2]['event_label'].long(), data[3].long()
        class_output, domain_output = self.forward(text, image, mask)
        class_loss = self.loss_funcs[0](class_output,
                                        label) * self.loss_weights[0]
        domain_loss = self.loss_funcs[1](domain_output,
                                         event_label) * self.loss_weights[1]
        loss = class_loss + domain_loss

        msg = f'class_loss={class_loss}, domain_loss={domain_loss}'
        return loss, msg

    def predict(self, data_without_label) -> torch.Tensor:
        text, mask, image = data_without_label[0][0], data_without_label[0][
            1], data_without_label[1]
        class_output, _ = self.forward(text, image, mask)
        return class_output
