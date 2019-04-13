""" 
    File Name:          MoReL/multi_head_attn.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask=None,
              dropout: nn.Module = None):

    # The shape of scores is
    # (batch_size, num_heads, seq_length, seq_length)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(query.size(-1))

    if mask is not None:
        mask = mask.unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    # The shape of scores is
    # (batch_size, num_heads, seq_length, emb_dim_per_head)
    scores = torch.matmul(scores, value)
    return scores


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 dropout: float = 0.0):

        super().__init__()

        self.__emb_dim = emb_dim
        self.__num_heads = num_heads
        self.__emb_dim_per_head = emb_dim // num_heads

        self.__q_linear = nn.Linear(emb_dim, emb_dim)
        self.__v_linear = nn.Linear(emb_dim, emb_dim)
        self.__k_linear = nn.Linear(emb_dim, emb_dim)

        self.__dropout = nn.Dropout(dropout)

        # Output layer of each attention cell
        self.__out_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v, mask=None):

        # Input key, value, and query are all in the shape of
        # (batch_size, num_heads, emb_dim)

        batch_size = q.size(0)

        # Shape of (batch_size, seq_length, num_heads, emb_dim_per_head)
        shape = (batch_size, -1, self.__num_heads, self.__emb_dim_per_head)

        q = self.__q_linear(q).view(*shape).transpose(1, 2)
        k = self.__k_linear(k).view(*shape).transpose(1, 2)
        v = self.__v_linear(v).view(*shape).transpose(1, 2)

        # Key, value, and query are all in the shape of
        # (batch_size, num_heads, seq_length, emb_dim_per_head)

        # Scores of attention have shape
        # (batch_size, num_heads, seq_length, emb_dim_per_head)
        scores = attention(query=q, key=k, value=v,
                           mask=mask, dropout=self.__dropout)

        # Concatenate heads (as input of the last layer)
        # (batch_size, seq_length, emb_dim)
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, -1, self.__emb_dim)

        return self.__out_linear(concat)
