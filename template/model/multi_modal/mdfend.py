"""
MDFEND: Multi-domain Fake News Detection
paper: https://arxiv.org/pdf/2201.00987
code: https://github.com/kennqiang/MDFEND-Weibo21
"""
from typing import List, Optional, Tuple, Callable

import torch
from torch import nn
from torch import Tensor
from transformers import BertModel

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
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class _CNNLayer(nn.Module):
    # def __init__(self, input_size: int, feature_kernel: Dict[int, int]):
    #     """
    #     Args:
    #         feature_kernel: {kernel_size: out_channels}
    #     """
    #     super(_CNNLayer, self).__init__()
    #     self.convs = torch.nn.ModuleList(
    #         [torch.nn.Conv1d(input_size, out_channels=feature_num, kernel_size=kernel)
    #          for kernel, feature_num in feature_kernel.items()])
    #     input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
    def __init__(self, input_size: int, filter_num: int,
                 filter_size: List[int]):
        """
        Args:
        """
        super(_CNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(input_size, out_channels=filter_num, kernel_size=k)
            for k in filter_size
        ])

    def forward(self, input_data: Tensor) -> Tensor:
        # todo permute
        share_input_data = input_data.permute(0, 2, 1)
        features = [conv(share_input_data) for conv in self.convs]
        features = [
            torch.max_pool1d(feature, feature.shape[-1])
            for feature in features
        ]
        features = torch.cat(features, dim=1)
        features = features.view([-1, features.shape[1]])
        return features


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
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class MDFEND(AbstractModel):
    def __init__(self,
                 embedding_size: int,
                 pre_trained_bert_name: str,
                 mlp_dims: List[int],
                 dropout_rate: float,
                 domain_num,
                 expert_num=5,
                 loss_func: Callable = nn.BCELoss()):
        super(MDFEND, self).__init__()
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.bert = BertModel.from_pretrained(
            pre_trained_bert_name).requires_grad_(False)
        self.loss_func = loss_func

        # key: kernel_size, value: out_channels
        # todo 实验更改out channels不一致
        # feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        # experts = [_CNNLayer(embedding_size, feature_kernel) for _ in range(self.expert_num)]
        filter_num = 64
        filter_size = [1, 2, 3, 5, 10]
        experts = [
            _CNNLayer(embedding_size, filter_num, filter_size)
            for _ in range(self.expert_num)
        ]
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(nn.Linear(embedding_size * 2, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.expert_num),
                                  nn.Softmax(dim=1))

        self.attention = _MaskAttentionLayer(embedding_size)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num,
                                            embedding_dim=embedding_size)
        self.classifier = _MLP(320, mlp_dims, dropout_rate)

    def forward(self, text: Tensor, mask: Tensor, domain: Tensor):
        text_embedding = self.bert(text, attention_mask=mask).last_hidden_state
        attention_feature, _ = self.attention(text_embedding, mask)

        # todo view(-1, 1)
        domain_embedding = self.domain_embedder(domain.view(-1, 1)).squeeze(1)

        gate_input = torch.cat([domain_embedding, attention_feature], dim=-1)
        gate_output = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.expert_num):
            expert_feature = self.experts[i](text_embedding)
            shared_feature += (expert_feature * gate_output[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        # after sigmoid: shape=(n,), data type=float
        return torch.sigmoid(label_pred.squeeze(1))

    def calculate_loss(self, data):
        token_ids, masks, domains, labels = data
        output = self.forward(token_ids, masks, domains)
        return self.loss_func(output, labels.float())

    def predict(self, data_without_label):
        token_ids, masks, domains = data_without_label

        # shape=(n,), data = 1 or 0
        round_pred = torch.round(
            self.forward(token_ids, masks, domains)).long()
        # after one hot: shape=(n,2), data = [0,1] or [1,0]
        one_hot_pred = torch.nn.functional.one_hot(round_pred)
        return one_hot_pred
