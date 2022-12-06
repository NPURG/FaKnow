from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function
from torchvision.models import VGG19_Weights

from template.model.model import AbstractModel


class _ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 只需要对输入的x返回loss，其他的返回None
        # 详见 https://zhuanlan.zhihu.com/p/263827804
        return grad_output * -ctx.lambd, None


class EANN(AbstractModel):
    def __init__(self, event_num: int, hidden_size: int,
                 dropout: int, reverse_lambd: int, embed_dim: int, embed_weight, vocab_size):
        super(EANN, self).__init__()
        # self.args = args

        self.event_num = event_num
        # self.embed_dim = embed_dim
        self.embed_dim = embed_weight[0].shape[0]
        self.hidden_size = hidden_size
        # self.lstm_size = self.embed_dim
        self.reverse_lambd = reverse_lambd

        # TEXT RNN
        # 真正的word embedding，使用了预训练好的权重
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embed_weight))
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(self.embed_dim, self.hidden_size)

        # TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([
            nn.Conv2d(channel_in, filter_num, (K, self.embed_dim))
            for K in window_size
        ])
        self.text_ccn_fc = nn.Linear(
            len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        for param in vgg_19.parameters():
            param.requires_grad = False

        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc = nn.Linear(num_ftrs, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

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
        text = text.unsqueeze(1)
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs
                ]  # [(N,hidden_dim,W), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.text_ccn_fc(text))

        # combine Text and Image
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)
        # Domain (which Event)
        reverse_feature = _ReverseLayer.apply(text_image, self.reverse_lambd)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def init_hidden(self, batch_size):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.lstm_size, requires_grad=True),
                torch.zeros(1, batch_size, self.lstm_size, requires_grad=True))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def calculate_loss(self):
        pass

    def predict(self, data):
        pass
