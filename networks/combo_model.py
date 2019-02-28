""" 
    File Name:          MoReL/combo_model.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/25/19
    Python Version:     3.5.4
    File Description:   
        This file contains that model for molecular representation learning.
"""
import torch.nn as nn

import utils.data_prep.config as c
from argparse import Namespace
from networks.common.reshape import Reshape
from networks.gcn.gcn import GCN
from networks.ggnn.ggnn import GGNN
from networks.transformer.encoder import Encoder


class Morel(nn.Module):

    def __init__(self, args: Namespace):

        super().__init__()

        if args.model_type.lower() == 'xfmr':
            self.__encoder = nn.Sequential(
                Encoder(dict_size=len(c.SMILES_TOKEN_DICT),
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
                        enc_dropout=args.xfmr_enc_dropout),
                nn.Linear(in_features=args.xfmr_emb_dim,
                          out_features=1))

        elif args.model_type.lower() == 'ggnn':
            self.__encoder = nn.Sequential(
                GGNN(state_dim=args.ggnn_state_dim,
                     num_nodes=c.MAX_NUM_ATOMS,
                     num_edge_types=len(c.BOND_FEAT_FUNC_LIST) + 1,
                     annotation_dim=len(c.ATOM_FEAT_FUNC_LIST),
                     propagation_steps=args.ggnn_propagation_steps),
                Reshape(),
                nn.Linear(in_features=(c.MAX_NUM_ATOMS * args.ggnn_state_dim),
                          out_features=1))

        elif args.model_type.lower() == 'gcn':
            self.__encoder = nn.Sequential(
                GCN(state_dim=args.gcn_state_dim,
                    num_nodes=c.MAX_NUM_ATOMS,
                    num_edge_types=len(c.BOND_FEAT_FUNC_LIST) + 1,
                    annotation_dim=len(c.ATOM_FEAT_FUNC_LIST)),
                Reshape(),
                nn.Linear(in_features=(c.MAX_NUM_ATOMS * args.gcn_state_dim),
                          out_features=1))

        elif args.model_type.lower() == 'cnn':
            # TODO: 1-D CNN encoder here
            # Possible param:
            # * Number of layers (args.cnn_num_layers)
            # *
            pass

        elif args.model_type.lower() == 'rnn':
            # TODO: RNN encoder here
            pass

        else:
            # TODO: simple dense encoder here
            pass

