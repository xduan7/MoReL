""" 
    File Name:          MoReL/mpnn.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/10/19
    Python Version:     3.5.4
    File Description:

        Message Passing Neural Network based on:
            https://arxiv.org/pdf/1704.01212.pdf

        This model is highly based on the QM9 example in PyG:
            https://github.com/rusty1s/pytorch_geometric/blob/master/
            examples/qm9_nn_conv.py

"""
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
from torch.nn import Sequential, Linear, ReLU, GRU


class MPNN(nn.Module):
    """
    Message Passing Neural Network
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 64,
                 num_conv: int = 3,
                 out_dim: int = 1):

        super(MPNN, self).__init__()

        self.__in_linear = nn.Sequential(
            nn.Linear(node_attr_dim, state_dim),
            ReLU())

        self.__num_conv = num_conv
        self.__nn_conv_linear = Sequential(
            Linear(edge_attr_dim, state_dim),
            ReLU(),
            Linear(state_dim, state_dim * state_dim))
        self.__nn_conv = pyg_nn.NNConv(
            state_dim, state_dim, self.__nn_conv_linear,
            aggr='mean', root_weight=False)

        self.__gru = GRU(state_dim, state_dim)
        self.__set2set = pyg_nn.Set2Set(state_dim, processing_steps=3)

        self.__out_linear = nn.Sequential(
            Linear(2 * state_dim, state_dim),
            ReLU(),
            Linear(state_dim, out_dim))

    def forward(self, data: pyg_data.Data):

        out = self.__in_linear(data.x)

        h = out.unsqueeze(0)

        for i in range(self.__num_conv):
            m = F.relu(self.__nn_conv(out, data.edge_index, data.edge_attr))
            out, h = self.__gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.__set2set(out, data.batch)
        out = self.__out_linear(out)
        return out.view(-1)

