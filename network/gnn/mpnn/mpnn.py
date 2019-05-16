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


class MPNN(nn.Module):
    """
    Message Passing Neural Network
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 64,
                 num_conv: int = 3,
                 out_dim: int = 1,
                 attention_pooling: bool = False):

        super(MPNN, self).__init__()

        self.__in_linear = nn.Sequential(
            nn.Linear(node_attr_dim, state_dim),
            nn.ReLU())

        self.__num_conv = num_conv
        self.__nn_conv_linear = nn.Sequential(
            nn.Linear(edge_attr_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim * state_dim))
        self.__nn_conv = pyg_nn.NNConv(
            state_dim, state_dim, self.__nn_conv_linear,
            aggr='mean', root_weight=False)
        self.__gru = nn.GRU(state_dim, state_dim)

        # self.__set2set = pyg_nn.Set2Set(state_dim, processing_steps=3)
        if attention_pooling:
            self.__pooling = pyg_nn.GlobalAttention(
                nn.Linear(state_dim, 1),
                nn.Linear(state_dim, 2 * state_dim))
        else:
            # Setting the num_layers > 1 will take significantly more time
            self.__pooling = pyg_nn.Set2Set(state_dim,
                                            processing_steps=3)

        self.__out_linear = nn.Sequential(
            nn.Linear(2 * state_dim, 2 * state_dim),
            nn.ReLU(),
            nn.Linear(2 * state_dim, out_dim))

    def forward(self, data: pyg_data.Data):

        out = self.__in_linear(data.x)
        h = out.unsqueeze(0)

        # Now out has the shape of [num_nodes, state_dim],
        # and h has the shape of [1, num_nodes, state_dim]

        for i in range(self.__num_conv):
            m = F.relu(self.__nn_conv(out, data.edge_index, data.edge_attr))
            out, h = self.__gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # Note that data.bach has the shape of [num_nodes]
        # which specifies the node's graph id in a batch
        out = self.__pooling(out, data.batch)

        # Now out is of size [batch_size, 2 * state_dim]
        return self.__out_linear(out)
        # return out.view(-1)
