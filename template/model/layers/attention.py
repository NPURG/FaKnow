from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

"""
layers for attention mechanism, including:
Feed-Forward Networks
Scaled Dot Product TransformerBlock
Mutil-head Attention
"""


def sequence_mask(X: Tensor, valid_len: Optional[Tensor] = None, value=0.):
    """Mask irrelevant entries in sequences."""
    max_len = X.size(1)
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X: Tensor, valid_lens: Optional[Tensor] = None):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int,
                 ffn_num_outputs: int, dropout=0., activation=F.relu):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.activation = activation
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense2(self.activation(self.dense1(x)))
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0., epsilon=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.epsilon = epsilon
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    # Shape of queries: (batch_size, num_queries, d)
    # Shape of keys: (batch_size, num_key-value pairs, d)
    # Shape of values: (batch_size, num_key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, num_queries)
    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                valid_lens: Optional[Tensor] = None):
        d = queries.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of keys
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
                 out_size:  Optional[int] = None,
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
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)

        # Shape of valid_lens:
        # (batch_size,) or (batch_size, no. of queries)

        # After transpose:
        # (batch_size * head_num, no. of queries or key-value pairs, num_hiddens/head_num)
        queries = transpose_qkv(self.W_q(queries), self.head_num)
        keys = transpose_qkv(self.W_k(keys), self.head_num)
        values = transpose_qkv(self.W_v(values), self.head_num)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # head_num times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.head_num,
                                                 dim=0)

        # Shape of output: (batch_size * head_num, no. of queries, num_hiddens/head_num)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = transpose_output(output, self.head_num)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""

    # before reshape:
    # (batch_size, no. of queries or key-value pairs, num_hiddens).
    # after reshape:
    # (batch_size, no. of queries or key-value pairs, head_num, num_hiddens/head_num)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output X:
    # (batch_size, head_num, no. of queries or key-value pairs, num_hiddens/head_num)
    X = X.permute(0, 2, 1, 3)

    # Shape of output:
    # (batch_size * head_num, no. of queries or key-value pairs, num_hiddens/head_num)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of transpose_qkv."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
