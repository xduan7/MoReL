""" 
    File Name:          MoReL/ggnn.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:

        Some references:

        https://github.com/JamesChuanggg/ggnn.pytorch/
        https://github.com/KaihuaTang/GGNN-for-bAbI-dataset.pytorch.1.0/

        The implementation in this file is different from references.
"""
import torch
import torch.nn as nn
from network.gnn.ggnn.propagator import Propagator


class GGNN(nn.Module):
    """
    TODO: dropout in this model (in the linear layers)

    """
    def __init__(self,
                 state_dim: int,
                 num_nodes: int,
                 num_edge_types: int,
                 annotation_dim: int,
                 propagation_steps: int,
                 directional_edges: bool = False):

        super().__init__()

        self.__state_dim = state_dim
        self.__num_nodes = num_nodes
        self.__num_edge_types = num_edge_types
        self.__annotation_dim = annotation_dim
        self.__propagation_steps = propagation_steps
        self.__directional_edges = directional_edges

        # Fully connect layers for in-going and out-going graph edges
        self.__linear_in = nn.Linear(self.__state_dim,
                                     self.__state_dim * self.__num_edge_types)
        self.__linear_out = nn.Linear(self.__state_dim,
                                      self.__state_dim * self.__num_edge_types)

        self.__propagator = Propagator(
            self.__state_dim, self.__num_nodes,
            self.__num_edge_types)

        self.__output = nn.Sequential(
            nn.Linear(self.__state_dim + self.__annotation_dim,
                      self.__state_dim))
            # nn.Tanh(),
            # nn.Linear(self.__state_dim, 1))

    def __state_reshape(self,
                        states):
        # Matrix state (either in-going or out-going) has size
        # [batch_size, num_node, state_dim * num_edge_types]
        states_ = states.view(
            -1, self.__num_nodes, self.__state_dim, self.__num_edge_types)
        # [batch_size, num_nodes, state_dim, num_edge_types]
        states_ = states_.transpose(2, 3).transpose(1, 2).contiguous()
        # [batch_size * num_edge_types * num_nodes * state_dim]
        return states_.view(
            -1, self.__num_nodes * self.__num_edge_types, self.__state_dim)

    def forward(self,
                init_state,
                annotation,
                adj_matrix):
        """
        :param init_state:
            [batch_size, num_nodes, state_dim]
        :param annotation:
            [batch_size, num_nodes, annotation_dim]
        :param adj_matrix:
            if edges are directional:
                [batch_size, num_nodes, num_nodes, num_edge_types * 2]
            else:
                [batch_size, num_nodes, num_nodes, num_edge_types]
        :return:
            [batch_size, num_node, state_dim]
        """

        curr_state = init_state
        for i_step in range(self.__propagation_steps):

            # Matrices in_states, out_states size:
            # [batch_size, num_nodes * num_edge_types, state_dim]
            in_states = self.__state_reshape(self.__linear_in(curr_state))
            out_states = self.__state_reshape(self.__linear_out(curr_state))

            curr_state = self.__propagator(
                in_states, out_states, curr_state, adj_matrix)

        # Equation (7) in section 3.3
        # Return a vector of size [batch_size, num_node, state_dim]
        # TODO: need to check the shape here and see if .sum(-1) is needed
        return self.__output(torch.cat((curr_state, annotation), -1)).sum(-1)
