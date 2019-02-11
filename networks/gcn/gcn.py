""" 
    File Name:          MoReL/gcn.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

        reference implementation maintained by GCN author
        https://github.com/tkipf/pygcn
"""
import torch.nn as nn
import torch.nn.functional as F

from networks.gcn.graph_conv_layer import GraphConvLayer


class GCN(nn.Module):
    """
    TODO: dropout in this model (between conv layers)

    """
    def __init__(self,
                 state_dim: int,
                 num_nodes: int,
                 num_edge_types: int,
                 annotation_dim: int):

        super().__init__()

        self.__model = nn.Sequential(
            GraphConvLayer(in_state_dim=annotation_dim,
                           out_state_dim=state_dim,
                           num_nodes=num_nodes,
                           num_edge_types=num_edge_types),

            nn.ReLU(inplace=True),
            # Dropout here?

            GraphConvLayer(in_state_dim=state_dim,
                           out_state_dim=state_dim,
                           num_nodes=num_nodes,
                           num_edge_types=num_edge_types))

    def forward(self,
                annotation,
                adj_matrix):

        return self.__model(annotation, adj_matrix)

