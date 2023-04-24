from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

"""
layers for transformer, including:
Feed-Forward Networks
AddNorm
Scaled Dot Product Attention
Mutil-head Attention
Encoder Layer
"""


def sequence_mask(x: Tensor, valid_len: Optional[Tensor] = None, value=0.):
    """Mask irrelevant entries in sequences."""
    max_len = x.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(x: Tensor, valid_lens: Optional[Tensor] = None):
    """Perform softmax operation by masking elements on the last axis."""
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


def transpose_qkv(x, num_heads):
    """
    Transposition for parallel computation of multiple attention heads.
    Args:
        x: (batch_size, num, num_hiddens), num_hiddens = head_num * out_size
        num_heads: number of attention heads

    Returns:
        (batch_size * head_num, num, num_hiddens/head_num)
    """

    # after:
    # (batch_size, num, head_num, num_hiddens/head_num)
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)

    # after:
    # (batch_size, head_num, num, num_hiddens/head_num)
    x = x.permute(0, 2, 1, 3)

    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    """
    Reverse the operation of transpose_qkv.
    Args:
        x: (batch_size * head_num, num, num_hiddens/head_num)
        num_heads: number of attention heads

    Returns:
        (batch_size, num, num_hiddens), num_hiddens = head_num * out_size
    """
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class FFN(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 dropout=0.,
                 activation=F.relu):
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
    def __init__(self, normalized_shape: int, dropout=0.):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: Tensor, y: Tensor):
        """
        Args:
            x: residual
            y: output of sublayer
        """
        return self.ln(self.dropout(y) + x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0., epsilon=0.):
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
            queries: (batch_size, q_num, d)
            keys: (batch_size, k-v_num, d)
            values: (batch_size, k-v_num, v-dim)
            valid_lens: (batch_size,) or (batch_size, q_num)
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(
            1, 2)) / (d ** 0.5 + self.epsilon)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
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
        self.W_o = nn.Linear(v_out_size * head_num, self.out_size * head_num, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries: (batch_size, q_num, d)
            keys: (batch_size, k-v_num, d)
            values: (batch_size, k-v_num, v-dim)
            valid_lens: (batch_size,) or (batch_size, q_num)
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

        # (batch_size * head_num, num, out_size)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of output_concat: (batch_size, num, head_num * out_size
        output_concat = transpose_output(output, self.head_num)
        return self.W_o(output_concat)


class EncoderLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 ffn_hidden_size: int,
                 head_num: int,
                 k_out_size: int,
                 v_out_size: int,
                 dropout=0.,
                 bias=False):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(input_size,
                                            k_out_size,
                                            v_out_size,
                                            head_num,
                                            dropout=dropout,
                                            bias=bias)
        self.addnorm1 = AddNorm(input_size, dropout)
        self.ffn = FFN(input_size,
                       ffn_hidden_size,
                       input_size,
                       dropout)
        self.addnorm2 = AddNorm(input_size, dropout)

    def forward(self, x, valid_lens=None):
        y = self.addnorm1(x, self.attention(x, x, x, valid_lens))
        return self.addnorm2(y, self.ffn(y))