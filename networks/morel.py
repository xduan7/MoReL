""" 
    File Name:          MoReL/morel.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/25/19
    Python Version:     3.5.4
    File Description:   
        This file contains that model for molecular representation learning.
"""
import argparse
import torch.nn as nn

import utils.data_prep.config as c
from networks.ggnn.ggnn import GGNN
from networks.transformer.encoder import Encoder


class Morel(nn.Module):

    def __init__(self, args: argparse.Namespace):

        super().__init__()

        if args.model_type.lower() == 'xfmr':
            self.__encoder = Encoder(
                dict_size=len(c.SMILES_TOKEN_DICT),
                seq_length=c.MAX_LEN_TOKENIZED_SMILES,
                base_feq=args.xfmr_base_freq,
                emb_scale=args.xfmr_emb_scale,
                emb_dim=args.xfmr_emb_dim,
                num_layers=args.xfmr_num_layers,
                num_heads=args.xfmr_num_heads,
                ff_mid_dim=args.xfmr_ff_mid_dim,
                pe_dropout=args.xfmr_pe_dropout,
                mha_dropout=args.xfmr_mha_dropout,
                ff_dropout=args.xfmr_ff_dropout,
                enc_dropout=args.xfmr_enc_dropout)

        elif args.model_type.lower() == 'ggnn':
            self.__encoder = GGNN(
                state_dim=args.ggnn_state_dim,
                num_nodes=c.MAX_NUM_ATOMS,
                num_edge_types=len(c.BOND_FEAT_FUNC_LIST) + 1,
                annotation_dim=len(c.ATOM_FEAT_FUNC_LIST),
                propagation_steps=args.ggnn_propagation_steps)

        else:
            # TODO: GCN encoder
            # TODO: simple dense, CNN, and RNN encoders
            pass
