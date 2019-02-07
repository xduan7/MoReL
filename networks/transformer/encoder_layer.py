""" 
    File Name:          MoReL/encoder_layer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn

from networks.transformer.feed_forward import FeedForward
from networks.transformer.multi_head_attn import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Each encoder layer:
    x -> Multi-head Attention -> Add & Norm -> Feed Forward -> Add & Norm -> y
    """
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 ff_mid_dim: int = 2048,
                 mha_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 enc_dropout: float = 0.0,
                 epsilon: float = 1e-6):

        super().__init__()

        self.__norm_for_mha = nn.LayerNorm(normalized_shape=emb_dim,
                                           eps=epsilon)
        self.__dropout_for_mha = nn.Dropout(enc_dropout)
        self.__mha = MultiHeadAttention(emb_dim=emb_dim,
                                        num_heads=num_heads,
                                        dropout=mha_dropout)

        self.__norm_for_ff = nn.LayerNorm(normalized_shape=emb_dim,
                                          eps=epsilon)
        self.__dropout_for_ff = nn.Dropout(enc_dropout)
        self.__ff = FeedForward(emb_dim=emb_dim,
                                mid_dim=ff_mid_dim,
                                dropout=ff_dropout)

    def forward(self, x, mask=None):

        norm_x = self.__norm_for_mha(x)

        h = x + self.__dropout_for_mha(
            self.__mha(norm_x, norm_x, norm_x, mask))

        norm_h = self.__norm_for_ff(h)

        return h + self.__dropout_for_ff(self.__ff(norm_h))
