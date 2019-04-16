""" 
    File Name:          MoReL/__graph_conv_layer.py
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

from utils.misc.sparse_tensor_helper import to_dense


def gcn_state_reshape(state: torch.Tensor,
                      num_nodes: int,
                      in_state_dim: int,
                      num_edge_types: int):
    # Matrix state (either in-going or out-going) has size
    # [batch_size, num_node, state_dim * num_edge_types]
    state_ = state.view(-1, num_nodes, in_state_dim, num_edge_types)
    # [batch_size, num_nodes, state_dim, num_edge_types]
    state_ = state_.transpose(2, 3).transpose(1, 2).contiguous()
    # [batch_size * num_edge_types * num_nodes * state_dim]
    return state_.view(-1, num_nodes * num_edge_types, in_state_dim)


class GraphConvLayer(nn.Module):

    def __init__(self,
                 in_state_dim: int,
                 out_state_dim: int,
                 num_nodes: int,
                 num_edge_types: int,
                 use_bias: bool = True):

        super().__init__()

        self.__in_state_dim = in_state_dim
        self.__out_state_dim = out_state_dim
        self.__num_nodes = num_nodes
        self.__num_edge_types = num_edge_types

        self.__linear_in = nn.Linear(self.__in_state_dim,
                                     self.__in_state_dim *
                                     self.__num_edge_types, bias=use_bias)
        self.__linear_out = nn.Linear(self.__in_state_dim,
                                      self.__out_state_dim, bias=use_bias)

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
        tmp_state = gcn_state_reshape(self.__linear_in(in_state),
                                      self.__num_nodes,
                                      self.__in_state_dim,
                                      self.__num_edge_types)

        # [batch_size, num_nodes, in_state_dim]
        tmp_adj_matrix = to_dense(adj_matrix).view(
            -1, self.__num_nodes, self.__num_nodes * self.__num_edge_types)

        # Note that bmm does not support sparse matrix in 1.0.1
        tmp_state = torch.bmm(tmp_adj_matrix, tmp_state)

        return self.__linear_out(tmp_state)


# Test out the correctness of graph convolution layer
if __name__ == '__main__':

    emb_dim = 5
    in_dim = out_dim = emb_dim
    batch_size = 1

    n_nodes = 4
    n_edge_types = 1

    _in_state = torch.rand(batch_size, n_nodes, emb_dim)
    _adj_matrix = torch.eye(n_nodes).view(
        batch_size, n_nodes, n_nodes, n_edge_types)

    # Connect nodes 0 and 1
    _adj_matrix[0, 0, 1, 0] = 1
    _adj_matrix[0, 1, 0, 0] = 1

    # Check if everything works when adjacency matrix is sparse

    # Print out the intermediate results (tmp_state)
    # To make sure that features will only be used/shared between neighbors
    print(_in_state)

    t_adj_matrix = to_dense(_adj_matrix).view(
        1, n_nodes, n_nodes * n_edge_types)
    t_state = torch.bmm(t_adj_matrix, _in_state)

    print(t_state)

    # Assert the nodes attributes are unchanged except for connected ones
    assert torch.all(torch.eq(_in_state[:batch_size, 2:, :emb_dim],
                              t_state[:batch_size, 2:, :emb_dim]))

