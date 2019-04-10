""" 
    File Name:          MoReL/propagator.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import torch.nn as nn

from utils.misc.sparse_tensor_helper import to_dense


class Propagator(nn.Module):

    def __init__(self,
                 state_dim: int,
                 num_nodes: int,
                 num_edge_types: int,
                 directional_edges: bool = False):

        super().__init__()

        self.__num_nodes = num_nodes
        self.__num_edge_types = num_edge_types
        self.__directional_edges = directional_edges

        # Uses the same names as PyTorch GRUs
        # pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#GRU
        self.__reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid())

        self.__update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid())

        self.__new_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh())

    def forward(self,
                in_states,
                out_states,
                curr_state,
                adj_matrix):
        """
        :param in_states:
            [batch_size, num_nodes * num_edge_types, state_dim]
        :param out_states:
            [batch_size, num_nodes * num_edge_types, state_dim]
        :param curr_state:
            [batch_size, num_node, state_dim]
        :param adj_matrix:
            if edges are directional:
                [batch_size, num_nodes, num_nodes, num_edge_types * 2]
            else:
                [batch_size, num_nodes, num_nodes, num_edge_types]
        :return:
        """

        # Divide the adjacency matrix into input edge
        # Matrices adj_matrix_in and adj_matrix_out size:
        # [batch_size, num_nodes, num_nodes * num_edge_types]
        matrix_size = self.__num_nodes * self.__num_edge_types
        t_adj_matrix = to_dense(adj_matrix)

        if self.__directional_edges:
            t_adj_matrix = t_adj_matrix.view(
                -1, self.__num_nodes, 2 * matrix_size)

            adj_matrix_in = t_adj_matrix[:, :, :matrix_size]
            adj_matrix_out = t_adj_matrix[:, :, matrix_size:]
        else:
            adj_matrix_in = adj_matrix_out = \
                t_adj_matrix.view(-1, self.__num_nodes, matrix_size)

        # Equation (2) in section 3.2
        # Note that we do not need the constant term (b) because it will be
        # taken care of in the reset_gate, update_gate, and transform
        # Matrices a_in and a_out size:
        # [batch_size, num_nodes, state_dim]
        # Matrix a size: [batch_size, num_nodes, state_dim * 2]
        a_in = torch.bmm(adj_matrix_in, in_states)
        a_out = torch.bmm(adj_matrix_out, out_states)
        a = torch.cat((a_in, a_out), -1)

        # Equation (3) and (4) in section 3.2
        # Matrix gate_input size: [batch_size, num_nodes, state_dim * 3]
        # Matrices r, z size: [batch_size, num_nodes, state_dim]
        gate_input = torch.cat((a, curr_state), -1)
        r = self.__reset_gate(gate_input)
        z = self.__update_gate(gate_input)

        # Equation (5) in section 3.2
        # Matrix h_hat size: [batch_size, num_nodes, state_dim]
        h_hat = self.__new_gate(torch.cat((a, r * curr_state), -1))

        # Returned matrix size: [batch_size, num_nodes, state_dim]
        return (1 - z) * curr_state + z * h_hat
