""" 
    File Name:          MoReL/gat.py
    Author:             Xiaotian Duan (xduan)
    Email:              xduan7@uchicago.edu
    Date:               4/16/2019
    Python Version:     3.5.4
    File Description:   

"""
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data


class GAT(nn.Module):

    def __init__(self,
                 node_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(GAT, self).__init__()

        __conv_layers = nn.ModuleList(
            [pyg_nn.GATConv(node_attr_dim, state_dim,
                            heads=num_heads, dropout=dropout_rate)])

        for i in range(num_conv - 1):
            __conv_layers.extend(
                [nn.ReLU(),
                 nn.Dropout(dropout_rate),
                 pyg_nn.GATConv(state_dim,
                                out_dim if (i == num_conv - 2) else state_dim,
                                heads=num_heads, dropout=dropout_rate)])

    def forward(self, data: pyg_data.Data):
        return self.__conv_layers(data.x, data.edge_index)


class EdgeGAT(nn.Module):
    """
    Version of GCN that takes one-hot encoded edge attribute
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 8,
                 num_heads: int = 8,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(EdgeGAT, self).__init__()

        self.__edge_attr_dim = edge_attr_dim
        __gat_kwargs = {
            'node_attr_dim': node_attr_dim,
            'state_dim': state_dim,
            'num_heads': num_heads,
            'num_conv': num_conv,
            'out_dim': out_dim,
            'dropout_rate': dropout_rate}

        self.__gat_nets = nn.ModuleList(
            [GAT(**__gat_kwargs) for _ in range(edge_attr_dim)])

    def forward(self, data: pyg_data.Data):

        out = []
        for i in range(self.__edge_attr_dim):
            # New graph that corresponds to the edge attributes
            _edge_index = torch.masked_select(
                data.edge_index, mask=data.edge_attr[:, i].byte()).view(2, -1)
            _data = pyg_nn.Data(x=data.x, edge_idnex=_edge_index)
            out.append(self.__gat_nets[i](_data))

        return torch.cat(tuple(out), dim=1)
