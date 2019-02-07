""" 
    File Name:          MoReL/transformer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn
import torch.nn.functional as F

from networks.transformer.decoder import Decoder
from networks.transformer.encoder import Encoder


class Transformer(nn.Module):
    # TODO: testing transfomrer (encoder has been tested)

    def __init__(self,
                 src_dict_size: int,
                 trg_dict_size: int,
                 seq_length: int,

                 base_feq: float,
                 emb_scale: float,

                 emb_dim: int,
                 num_layers: int,
                 num_heads: int,
                 ff_mid_dim: int,

                 pe_dropout: float = 0.0,
                 mha_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 enc_dropout: float = 0.0,

                 epsilon: float = 1e-6):

        super().__init__()

        network_kwargs = {
            'seq_length': seq_length,

            'base_feq': base_feq,
            'emb_scale': emb_scale,

            'emb_dim': emb_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ff_mid_dim': ff_mid_dim,

            'pe_dropout': pe_dropout,
            'mha_dropout': mha_dropout,
            'ff_dropout': ff_dropout,
            'enc_dropout': enc_dropout,

            'epsilon': epsilon, }

        self.__encoder = Encoder(dict_size=src_dict_size, **network_kwargs)
        self.__decoder = Decoder(dict_size=trg_dict_size, **network_kwargs)
        self.__output = nn.Linear(emb_dim, trg_dict_size)

    def forward(self,
                src_indexed_sentence,
                trg_indexed_sentence,
                src_mask,
                trg_mask):

        enc_out = self.__encoder(src_indexed_sentence, src_mask),
        h = self.__decoder(trg_indexed_sentence, enc_out, src_mask, trg_mask)
        return F.log_softmax(self.__output(h), dim=-1)

