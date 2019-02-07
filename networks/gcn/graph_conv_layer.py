""" 
    File Name:          MoReL/graph_conv_layer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

        reference from GCN author: https://github.com/tkipf/pygcn

        this implementation differs from his in the following way:
            * adding edge channels (features)
            * support mini-batch optimization
"""
import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):

    def __init__(self,
                 in_state_dim: int,
                 out_state_dim: int,
                 num_nodes: int,
                 num_edge_types: int,
                 bias: bool = True):

        super().__init__()

        self.__in_state_dim = in_state_dim
        self.__out_state_dim = out_state_dim
        self.__num_nodes = num_nodes
        self.__num_edge_types = num_edge_types

        self.__linear_in = nn.Linear(self.__in_state_dim,
                                     self.__in_state_dim *
                                     self.__num_edge_types)
        self.__linear_out = nn.Linear(self.__in_state_dim,
                                      self.__out_state_dim, bias=bias)

    def __state_reshape(self,
                        state,
                        state_dim):
        # Matrix state (either in-going or out-going) has size
        # [batch_size, num_node, state_dim * num_edge_types]
        state_ = state.view(
            -1, self.__num_nodes, state_dim, self.__num_edge_types)
        # [batch_size, num_nodes, state_dim, num_edge_types]
        state_ = state_.transpose(2, 3).transpose(1, 2).contiguous()
        # [batch_size * num_edge_types * num_nodes * state_dim]
        return state_.view(
            -1, self.__num_nodes * self.__num_edge_types, state_dim)

    def forward(self,
                in_state,
                adj_matrix):
        """
        :param in_state:
            [batch_size, num_nodes, in_dim]
        :param adj_matrix:
            [batch_size, num_nodes, num_nodes, num_edge_types]
        :return:
        """

        # [batch_size, num_edge_types * num_nodes, in_state_dim]
        tmp_state = self.__state_reshape(self.__linear_in(in_state),
                                         self.__in_state_dim)

        # [batch_size, num_nodes, in_state_dim]
        tmp_state = torch.bmm(
            adj_matrix.view(-1, self.__num_nodes,
                            self.__num_nodes * self.__num_edge_types),
            tmp_state)

        return self.__linear_out(tmp_state)
