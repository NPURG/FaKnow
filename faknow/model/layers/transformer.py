from typing import Optional, Union, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from zmq import device

"""
layers for transformer, including:
Feed-Forward Networks
Add&Norm
PositionalEncoding
Scaled Dot Product Attention
Mutil-head Attention
Encoder Layer
"""


def sequence_mask(x: Tensor, valid_len: Optional[Tensor] = None, value=0.):
    """
    Mask irrelevant entries in sequences.

    Args:
        x (Tensor): shape=(batch_size, num_steps, num_hiddens)
        valid_len (Tensor): shape=(batch_size,), default=None
        value (float): value to be substituted in masked entries, default=0.

    Returns:
        Tensor: masked input x, shape=(batch_size, num_steps)
    """

    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(x: Tensor, valid_lens: Optional[Tensor] = None):
    """
    Perform softmax operation by masking elements on the last axis.

    Args:
        x (Tensor): shape=(batch_size, num_steps, num_hiddens)
        valid_lens (Tensor): shape=(batch_size,), default=None

    Returns:
        Tensor: shape=(batch_size, num_steps, num_hiddens)
    """

    # x: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(x.reshape(shape), dim=-1)


def transpose_qkv(x: Tensor, num_heads: int):
    """
    Transposition for parallel computation of multiple attention heads.

    Args:
        x (Tensor): shape=(batch_size, num, num_hiddens),
            where num_hiddens = head_num * out_size
        num_heads (int): number of attention heads

    Returns:
        Tensor: shape=(batch_size * head_num, num, num_hiddens/head_num)
    """

    # after:
    # (batch_size, num, head_num, num_hiddens/head_num)
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)

    # after:
    # (batch_size, head_num, num, num_hiddens/head_num)
    x = x.permute(0, 2, 1, 3)

    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: Tensor, num_heads: int):
    """
    Reverse the operation of transpose_qkv.

    Args:
        x (Tensor): shape=(batch_size * head_num, num, num_hiddens/head_num)
        num_heads (int): number of attention heads

    Returns:
        Tensor: shape=(batch_size, num, num_hiddens),
            where num_hiddens = head_num * out_size
    """
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class FFN(nn.Module):
    """
    Feed-Forward Networks
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 dropout=0.,
                 activation: Optional[Callable] = F.relu):
        """
        Args:
            input_size(int): input dimension
            hidden_size(int): hidden layer dimension
            output_size(int): output dimension,
                if None, output_size=input_size, default=None
            dropout(float): dropout rate, default=0.
            activation(Callable): activation function, default=F.relu
        """
        super(FFN, self).__init__()

        if output_size is None:
            output_size = input_size

        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense2(self.activation(self.dense1(x)))
        return self.dropout(x)


class AddNorm(nn.Module):
    """
    residual add and layernorm
    """

    def __init__(self,
                 normalized_shape: Union[int, List[int], torch.Size],
                 dropout=0.):
        """
        Args:
            normalized_shape (Union[int, List[int], torch.Size]): input shape from an expected input of size
            dropout (float): dropout rate, default=0.
        """
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: Tensor, y: Tensor):
        """
        Args:
            x (Tensor): residual
            y (Tensor): output of sublayer

        Returns:
            Tensor: layernorm(x + dropout(y))
        """
        return self.ln(self.dropout(y) + x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for inputs of transformer
    """

    def __init__(self, dim: int, dropout=0., max_len=1000):
        """
        Args:
            dim(int): the embedding dimension of input.
            dropout(float): dropout rate, Default=0.
            max_len(int): the max length of sequence length, Default=1000.
        """

        super().__init__()
        self.pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        self.pe[:, 0::2] = torch.sin(position / div_term[0::2])
        self.pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, inputs: Tensor, step=None):
        """
        Args:
            inputs(Tensor):input tensor shape=(batch_size, length, embedding_dim)
            step(int): the cutting step of position encoding, Default=None

        Returns:
            Tensor: shape=(batch_size, length, embedding_dim)
        """
        if step is None:
            inputs = inputs + self.pe[:inputs.size(1), :].to(inputs.device)
        else:
            inputs = inputs + self.pe[:, step].to(inputs.device)
        inputs = self.dropout(inputs)
        return inputs


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention
    """

    def __init__(self, dropout=0., epsilon=0.):
        """
        Args:
            dropout (float): dropout rate, default=0.
            epsilon (float): small constant for numerical stability, default=0.
        """

        super(ScaledDotProductAttention, self).__init__()
        self.epsilon = epsilon
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                valid_lens: Optional[Tensor] = None):
        """
        Args:
            queries (Tensor): shape=(batch_size, q_num, d)
            keys (Tensor): shape=(batch_size, k-v_num, d)
            values (Tensor): shape=(batch_size, k-v_num, v_dim)
            valid_lens (Tensor): shape=(batch_size,) or (batch_size, q_num), default=None

        Returns:
            Tensor: attention_values, shape=(batch_size, q_num, v_dim)
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(
            1, 2)) / (d ** 0.5 + self.epsilon)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention with ScaledDotProductAttention
    """

    def __init__(self,
                 input_size: int,
                 k_out_size: int,
                 v_out_size: int,
                 head_num: int,
                 out_size: Optional[int] = None,
                 q_in_size: Optional[int] = None,
                 k_in_size: Optional[int] = None,
                 v_in_size: Optional[int] = None,
                 dropout=0.,
                 bias=False):
        """
        Args:
            input_size (int): input dimension
            k_out_size (int): output dimension of key
            v_out_size (int): output dimension of value
            head_num (int): number of attention heads
            out_size (int): output dimension,
                if None, out_size=input_size, default=None
            q_in_size (int): input dimension of query,
                if None, q_in_size=input_size, default=None
            k_in_size (int): input dimension of key,
                if None, k_in_size=input_size, default=None
            v_in_size (int): input dimension of value,
                if None, v_in_size=input_size, default=None
            dropout (float): dropout rate, default=0.
            bias (bool): whether to use bias in Linear layers, default=False
        """

        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.q_in_size = q_in_size if q_in_size is not None else input_size
        self.k_in_size = k_in_size if k_in_size is not None else input_size
        self.v_in_size = v_in_size if v_in_size is not None else input_size
        self.out_size = out_size if out_size is not None else input_size

        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(self.q_in_size, k_out_size * head_num, bias=bias)
        self.W_k = nn.Linear(self.k_in_size, k_out_size * head_num, bias=bias)
        self.W_v = nn.Linear(self.v_in_size, v_out_size * head_num, bias=bias)
        self.W_o = nn.Linear(v_out_size * head_num,
                             self.out_size * head_num,
                             bias=bias)

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                valid_lens: Optional[Tensor] = None):
        """
        Args:
            queries (Tensor): shape=(batch_size, q_num, d)
            keys (Tensor): shape=(batch_size, k-v_num, d)
            values (Tensor): shape=(batch_size, k-v_num, v-dim)
            valid_lens (Tensor): shape=(batch_size,) or (batch_size, q_num)

        Returns:
            Tensor: multi-head output, shape=(batch_size, q_num, out_size * head_num)
        """

        # After transpose:
        # (batch_size * head_num, num, out_size)
        queries = transpose_qkv(self.W_q(queries), self.head_num)
        keys = transpose_qkv(self.W_k(keys), self.head_num)
        values = transpose_qkv(self.W_v(values), self.head_num)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # head_num times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.head_num,
                                                 dim=0)

        # (batch_size * head_num, q_num, v_out_size)
        output = self.attention(queries, keys, values, valid_lens)

        # After transpose:
        # (batch_size, q_num, head_num * v_out_size)
        output_concat = transpose_output(output, self.head_num)
        return self.W_o(output_concat)


class EncoderLayer(nn.Module):
    """
    Encoder Layer in Transformer
    """

    def __init__(self,
                 input_size: int,
                 ffn_hidden_size: int,
                 head_num: int,
                 k_out_size: int,
                 v_out_size: int,
                 dropout=0.,
                 bias=False,
                 activation: Optional[Callable] = F.relu):
        """
        Args:
            input_size (int): input dimension
            ffn_hidden_size (int): hidden layer dimension of FFN
            head_num (int): number of attention heads
            k_out_size (int): output dimension of key
            v_out_size (int): output dimension of value
            dropout (float): dropout rate, default=0.
            bias (bool): whether to use bias in Linear layers, default=False
            activation(Callable): activation function for FFN, default=F.relu
        """

        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_size,
                                            k_out_size,
                                            v_out_size,
                                            head_num,
                                            dropout=dropout,
                                            bias=bias)
        self.addnorm1 = AddNorm(input_size, dropout)
        self.ffn = FFN(input_size, ffn_hidden_size, input_size, dropout, activation)
        self.addnorm2 = AddNorm(input_size, dropout)

    def forward(self, x: Tensor, valid_lens: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): shape=(batch_size, num_steps, input_size)
            valid_lens (Tensor): shape=(batch_size,), default=None

        Returns:
            Tensor: shape=(batch_size,) or (batch_size, q_num)
        """

        y = self.addnorm1(x, self.attention(x, x, x, valid_lens))
        return self.addnorm2(y, self.ffn(y))
