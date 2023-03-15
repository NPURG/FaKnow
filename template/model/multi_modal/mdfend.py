"""
MDFEND: Multi-domain Fake News Detection
paper: https://arxiv.org/pdf/2201.00987
code: https://github.com/kennqiang/MDFEND-Weibo21
"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from transformers import BertModel

from model.layers.layer import TextCNNLayer
from template.model.model import AbstractModel


class _MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dims: List[int],
                 dropout_rate: float,
                 output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): shared feature from domain and text, shape=(batch_size, embed_dim)

        """
        return self.mlp(x)


class _MaskAttentionLayer(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_size: int):
        super(_MaskAttentionLayer, self).__init__()
        self.attention_layer = torch.nn.Linear(input_size, 1)

    def forward(self,
                inputs: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weights = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1).unsqueeze(1)
        outputs = torch.matmul(weights, inputs).squeeze(1)
        return outputs, weights


class MDFEND(AbstractModel):
    r"""MDFEND: Multi-domain Fake News Detection
    """

    def __init__(self,
                 pre_trained_bert_name: str,
                 domain_num: int,
                 mlp_dims: Optional[List[int]] = None,
                 dropout_rate=0.2,
                 expert_num=5):
        """

        Args:
            pre_trained_bert_name (str): the name or local path of pre-trained bert model
            domain_num (int): total number of all domains
            mlp_dims (List[int]): a list of the dimensions in MLP layer. Default=[384]
            dropout_rate (float): rate of Dropout layer. Default=0.2
            expert_num (int): number of experts also called TextCNNLayer. Default=5
        """
        super(MDFEND, self).__init__()
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.bert = BertModel.from_pretrained(
            pre_trained_bert_name).requires_grad_(False)
        self.embedding_size = self.bert.config.hidden_size
        self.loss_func = nn.BCELoss()
        if mlp_dims is None:
            mlp_dims = [384]

        filter_num = 64
        filter_sizes = [1, 2, 3, 5, 10]
        experts = [
            TextCNNLayer(self.embedding_size, filter_num, filter_sizes)
            for _ in range(self.expert_num)
        ]
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, mlp_dims[-1]), nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.expert_num), nn.Softmax(dim=1))

        self.attention = _MaskAttentionLayer(self.embedding_size)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num,
                                            embedding_dim=self.embedding_size)
        self.classifier = _MLP(320, mlp_dims, dropout_rate)

    def forward(self, token_id: Tensor, mask: Tensor, domain: Tensor) -> Tensor:
        """

        Args:
            token_id (Tensor): token ids from bert tokenizer, shape=(batch_size, max_len)
            mask (Tensor): mask from bert tokenizer, shape=(batch_size, max_len)
            domain (Tensor): domain id, shape=(batch_size,)

        Returns:
            FloatTensor: the prediction of being fake, shape=(batch_size,)
        """
        text_embedding = self.bert(token_id, attention_mask=mask).last_hidden_state
        attention_feature, _ = self.attention(text_embedding, mask)

        domain_embedding = self.domain_embedder(domain.view(-1, 1)).squeeze(1)

        gate_input = torch.cat([domain_embedding, attention_feature], dim=-1)
        gate_output = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.expert_num):
            expert_feature = self.experts[i](text_embedding)
            shared_feature += (expert_feature * gate_output[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

    def calculate_loss(self, data) -> Tensor:
        token_ids, masks, domains, labels = data
        output = self.forward(token_ids, masks, domains)
        return self.loss_func(output, labels.float())

    def predict(self, data_without_label) -> Tensor:
        token_ids, masks, domains = data_without_label

        # shape=(n,), data = 1 or 0
        round_pred = torch.round(self.forward(token_ids, masks,
                                              domains)).long()
        # after one hot: shape=(n,2), data = [0,1] or [1,0]
        one_hot_pred = torch.nn.functional.one_hot(round_pred, num_classes=2)
        return one_hot_pred
