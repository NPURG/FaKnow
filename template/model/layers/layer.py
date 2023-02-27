from typing import Optional, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class TextCNNLayer(nn.Module):
    """
    It's not a full TextCNN model. Only convolution and max pooling are including here
    but without an embedding layer or fully connected layer. Thus, it should be a part of your own TextCNN model
    """
    def __init__(self,
                 embedding_dim: int,
                 filter_num: int,
                 filter_sizes: List[int],
                 activate_fn: Optional[Callable] = None):
        super().__init__()
        # in_channel=1, out_channel=filter_num
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_num, (k, embedding_dim)) for k in filter_sizes
        ])
        self.activate_fn = activate_fn

    def forward(self, embedded_text: torch.Tensor):
        # before unsqueeze: batch_size * max_len * embedding_dim
        embedded_text = embedded_text.unsqueeze(1)

        # before squeeze: batch_size * filter_num * (max_len-k+1) * 1
        if self.activate_fn is None:
            conv_features = [
                conv(embedded_text).squeeze(3) for conv in self.convs
            ]
        else:
            conv_features = [
                self.activate_fn(conv(embedded_text).squeeze(3))
                for conv in self.convs
            ]

        # before squeeze: batch_size * filter_num * 1
        # conv.shape[2] = (max_len-k+1)
        pool_features = [
            torch.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conv_features
        ]

        # batch_size * (filter_num * len(filter_sizes))
        concat_features = torch.cat(pool_features, dim=1)
        return concat_features


class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 只需要对输入的x返回loss，其他的返回None
        # 详见 https://zhuanlan.zhihu.com/p/263827804
        return grad_output * -ctx.lambd, None


class SignedAttention(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float,
                 alpha: float,
                 concat=True):
        super(SignedAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.fc_W = nn.Parameter(
            torch.zeros(size=(2 * out_features, out_features)))
        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.fc_W.data, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):

        h = torch.mm(x, self.W)
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leaky_relu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        negative_attention = torch.where(adj > 0, -e, zero_vec)
        negative_attention = -F.softmax(negative_attention, dim=1)
        negative_attention = F.dropout(negative_attention,
                                       self.dropout,
                                       training=self.training)
        h_prime = torch.matmul(attention, x)
        h_prime_negative = torch.matmul(negative_attention, x)
        h_prime_double = torch.cat([h_prime, h_prime_negative], dim=1)
        new_h_prime = torch.mm(h_prime_double, self.fc_W)
        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class SignedGAT(nn.Module):
    def __init__(self,
                 node_vectors: np.ndarray,
                 cos_sim_matrix: torch.Tensor,
                 num_features: int,
                 node_num: int,
                 adj_matrix: torch.Tensor,
                 head_num=4,
                 out_features=300,
                 dropout=0,
                 alpha=0.3):
        super(SignedGAT, self).__init__()
        self.dropout = dropout
        self.node_num = node_num
        self.node_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(node_vectors), padding_idx=0)
        self.original_adj = adj_matrix
        self.potential_adj = torch.where(cos_sim_matrix > 0.5,
                                         torch.ones_like(cos_sim_matrix),
                                         torch.zeros_like(cos_sim_matrix))
        self.adj = self.original_adj + self.potential_adj
        self.adj = torch.where(self.adj > 0, torch.ones_like(self.adj),
                               torch.zeros_like(self.adj))

        self.attentions = [
            SignedAttention(num_features,
                            out_features,
                            dropout=dropout,
                            alpha=alpha,
                            concat=True) for _ in range(head_num)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # multi head attention
        self.out_att = SignedAttention(num_features * head_num,
                                       out_features,
                                       dropout=dropout,
                                       alpha=alpha,
                                       concat=False)

    def forward(self, post_id: torch.Tensor):
        embedding = self.node_embedding(torch.arange(
            0, self.node_num).long()).to(torch.float32)
        x = F.dropout(embedding, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        return x[post_id]
