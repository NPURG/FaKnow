import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(
            torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp: torch.Tensor, adj: torch.Tensor):

        h = torch.mm(inp, self.W)
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention,
                                       self.dropout,
                                       training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime, h_prime_negative], dim=1)
        new_h_prime = torch.mm(h_prime_double, self.wtrans)
        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class SignedGAT(nn.Module):
    def __init__(
            self,
            node_embedding: np.ndarray,
            cos_sim_matrix: torch.Tensor,
            num_features: int,
            uV: int,
            original_adj: np.ndarray,
            num_heads=4, n_output=300, dropout=0, alpha=0.3,
            embedding_dim=300):
        super(SignedGAT, self).__init__()
        self.dropout = dropout
        self.uV = uV
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV,
                                                 embedding_dim=embedding_dim,
                                                 padding_idx=0)
        self.user_tweet_embedding.from_pretrained(
            torch.from_numpy(node_embedding))
        self.original_adj = torch.from_numpy(original_adj.astype(np.float64))
        self.potentinal_adj = torch.where(cos_sim_matrix > 0.5,
                                          torch.ones_like(cos_sim_matrix),
                                          torch.zeros_like(cos_sim_matrix))
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj > 0, torch.ones_like(self.adj),
                               torch.zeros_like(self.adj))

        self.attentions = [
            GraphAttentionLayer(num_features,
                                n_output,
                                dropout=dropout,
                                alpha=alpha,
                                concat=True) for _ in range(num_heads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(num_features * num_heads,
                                           n_output,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, tweet_id):
        embedding = self.user_tweet_embedding(torch.arange(0, self.uV).long()).to(
            torch.float32)
        x = F.dropout(embedding, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        return x[tweet_id]
