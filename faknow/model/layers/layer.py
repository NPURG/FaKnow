from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.autograd import Function
from transformers import BertModel


class TextCNNLayer(nn.Module):
    """
    It's not a whole TextCNN model. Only convolution and max pooling layers are
    included here but without an embedding layer or fully connected layer.
    Thus, it should be a part of your own TextCNN model.
    """

    def __init__(self,
                 embedding_dim: int,
                 filter_num: int,
                 filter_sizes: List[int],
                 activate_fn: Optional[Callable] = None):
        """
        Args:
            embedding_dim (int): the dimension of word embedding
            filter_num (int): the number of filters,
                which is also the output channel
            filter_sizes (List[int]): the size of filters
            activate_fn (Callable): the activation function of
                convolution layer. Default=None
        """

        super().__init__()
        # in_channel=1, out_channel=filter_num
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_num, (k, embedding_dim)) for k in filter_sizes
        ])
        self.activate_fn = activate_fn

    def forward(self, embedded_text: torch.Tensor):
        """
        Args:
            embedded_text (torch.Tensor): the embedded text,
                shape=(batch_size, max_len, embedding_dim)

        Returns:
            torch.Tensor: the output of convolution and max pooling layer,
                shape (batch_size, filter_num * len(filter_sizes))
        """

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

        # conv.shape[2] = (max_len - k + 1)
        # before squeeze: batch_size * filter_num * 1
        pool_features = [
            torch.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in conv_features
        ]

        # batch_size * (filter_num * len(filter_sizes))
        concat_features = torch.cat(pool_features, dim=1)
        return concat_features


class BertEncoder(nn.Module):
    """
    Text encoder based on BERT to encode text into vectors.
    """
    def __init__(self, bert: str, fine_tune=False):
        """
        Args:
            bert (str): the name of pretrained BERT model
            fine_tune (bool): whether to fine tune BERT or not, default=False
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert).requires_grad_(fine_tune)
        self.fine_tune = fine_tune
        self.dim = self.bert.config.hidden_size

    def forward(self, token_id: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            token_id (torch.Tensor): shape=(batch_size, max_len)
            mask (torch.Tensor): shape=(batch_size, max_len)

        Returns:
            torch.Tensor: last hidden state from bert, shape=(batch_size, max_len, dim)
        """
        return self.bert(token_id, attention_mask=mask).last_hidden_state


class ResNetEncoder(nn.Module):
    """
    Image encoder based on ResNet50 with pretrained weights on ImageNet1k
    to encode images pixels into vectors.
    """
    def __init__(self, out_size: int) -> None:
        """
        Args:
            out_size (int): the size of output features of the fc layer in ResNet
        """
        super().__init__()
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, out_size)

    def forward(self, image: Tensor) -> Tensor:
        """
        Args:
            image (torch.Tensor): image pixels, shape=(batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: output from pretrained resnet model, shape=(batch_size, out_size)
        """
        return self.resnet(image)


class GradientReverseLayer(Function):
    """
    gradient reverse layer,
    which is used to reverse the gradient in backward propagation,
    see https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    def forward(ctx, x, lambd):
        """
        Args:
            ctx (torch.autograd.function.Function): the context
            x (torch.Tensor): the input tensor
            lambd (float): the lambda value

        Returns:
            torch.Tensor: the input tensor x
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        reverse the gradient in backward propagation
        Args:
            ctx (torch.autograd.function.Function): the context
            grad_output (torch.Tensor): the gradient of output

        Returns:
            tuple:
                torch.Tensor: the reversed gradient
                None: None
        """
        return grad_output * -ctx.lambd, None


class SignedAttention(nn.Module):
    """
    signed attention layer for signed graph
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float,
                 alpha: float,
                 concat=True):
        """
        Args:
            in_features (int): the size of input features
            out_features (int): the size of output features
            dropout (float): the dropout rate
            alpha (float): the alpha value of LeakyReLU
            concat (bool): whether to concatenate the output features or not
        """

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
        """
        Args:
            x (torch.Tensor): the input features
            adj (torch.Tensor): the adjacency matrix

        Returns:
            torch.Tensor: the output features
        """

        h = torch.mm(x, self.W)
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leaky_relu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e, device=x.device)

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
    """
    signed graph attention network
    """

    def __init__(self,
                 node_vectors: torch.Tensor,
                 cos_sim_matrix: torch.Tensor,
                 num_features: int,
                 node_num: int,
                 adj_matrix: torch.Tensor,
                 head_num=4,
                 out_features=300,
                 dropout=0.,
                 alpha=0.3):
        """
        Args:
            node_vectors (torch.Tensor): the node vectors
            cos_sim_matrix (torch.Tensor): the cosine similarity matrix
            num_features (int): the size of input features
            node_num (int): the number of nodes
            adj_matrix (torch.Tensor): the adjacency matrix
            head_num (int): the number of attention heads
            out_features (int): the size of output features
            dropout (float): the dropout rate
            alpha (float): the alpha value of LeakyReLU
                in signed attention layer
        """

        super(SignedGAT, self).__init__()
        self.dropout = dropout
        self.node_num = node_num
        self.node_embedding = nn.Embedding.from_pretrained(node_vectors,
                                                           padding_idx=0)
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

        self.out_att = SignedAttention(num_features * head_num,
                                       out_features,
                                       dropout=dropout,
                                       alpha=alpha,
                                       concat=False)

    def forward(self, post_id: torch.Tensor):
        """
        Args:
            post_id (torch.Tensor): the post id

        Returns:
            torch.Tensor: the output features
        """

        embedding = self.node_embedding(torch.arange(
            0, self.node_num, device=post_id.device).long()).to(torch.float32)
        x = F.dropout(embedding, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        return x[post_id]
