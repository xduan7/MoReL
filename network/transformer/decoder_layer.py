""" 
    File Name:          MoReL/decoder_layer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn
from network.transformer.feed_forward import FeedForward
from network.transformer.multi_head_attn import MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    Decoder layer takes masked target input and encoder output
    """

    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 ff_mid_dim: int,
                 mha_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 enc_dropout: float = 0.0,
                 epsilon: float = 1e-6):

        super().__init__()

        # This part of the network takes masked target
        self.__norm_for_x_mha = nn.LayerNorm(normalized_shape=emb_dim,
                                             eps=epsilon)
        self.__dropout_for_x_mha = nn.Dropout(enc_dropout)
        self.__x_mha = MultiHeadAttention(emb_dim=emb_dim,
                                          num_heads=num_heads,
                                          dropout=mha_dropout)

        # This part of the network takes encoder layer output
        self.__norm_for_enc_mha = nn.LayerNorm(normalized_shape=emb_dim,
                                               eps=epsilon)
        self.__dropout_for_enc_mha = nn.Dropout(enc_dropout)
        self.__enc_mha = MultiHeadAttention(emb_dim=emb_dim,
                                            num_heads=num_heads,
                                            dropout=mha_dropout)

        # Feed-forward network
        self.__norm_for_ff = nn.LayerNorm(normalized_shape=emb_dim,
                                          eps=epsilon)
        self.__dropout_for_ff = nn.Dropout(enc_dropout)
        self.__ff = FeedForward(emb_dim=emb_dim,
                                mid_dim=ff_mid_dim,
                                dropout=ff_dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):

        norm_x = self.__norm_for_x_mha(x)

        h = x + self.__dropout_for_x_mha(
            self.__x_mha(norm_x, norm_x, norm_x, trg_mask))

        norm_h = self.__norm_for_enc_mha(h)

        h = h + self.__dropout_for_enc_mha(
            self.__enc_mha(norm_h, enc_out, enc_out, src_mask))

        norm_h = self.__norm_for_ff(h)

        return h + self.__dropout_for_ff(self.__ff(norm_h))
