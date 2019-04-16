""" 
    File Name:          MoReL/gcn.py
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


class GCN(nn.Module):

    def __init__(self,
                 node_attr_dim: int,
                 state_dim: int = 16,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(GCN, self).__init__()

        __conv_layers = nn.ModuleList([
            pyg_nn.GCNConv(node_attr_dim, state_dim, cached=True)])

        for i in range(num_conv - 1):
            __conv_layers.extend(
                [nn.ReLU(),
                 nn.Dropout(dropout_rate),
                 pyg_nn.GCNConv(state_dim,
                                out_dim if (i == num_conv - 2) else state_dim,
                                cached=True)])

        self.__conv_layers = nn.Sequential(__conv_layers)

        # Here we can either use global attention or set2set
        # Followed by some linear layers

    def forward(self, data: pyg_data.Data):
        return self.__conv_layers(data.x, data.edge_index)


class EdgeGCN(nn.Module):
    """
    Version of GCN that takes one-hot encoded edge attribute
    """

    def __init__(self,
                 node_attr_dim: int,
                 edge_attr_dim: int,
                 state_dim: int = 16,
                 num_conv: int = 2,
                 out_dim: int = 1,
                 dropout_rate: float = 0.2):

        super(EdgeGCN, self).__init__()

        self.__edge_attr_dim = edge_attr_dim
        __gcn_kwargs = {
            'node_attr_dim': node_attr_dim,
            'state_dim': state_dim,
            'num_conv': num_conv,
            'out_dim': out_dim,
            'dropout_rate': dropout_rate}

        self.__gcn_nets = nn.ModuleList(
            [GCN(**__gcn_kwargs) for _ in range(edge_attr_dim)])

    def forward(self, data: pyg_data.Data):

        out = []
        for i in self.__edge_attr_dim:
            # New graph that corresponds to the edge attributes
            _edge_index = torch.masked_select(
                data.edge_index, mask=data.edge_attr[:, i].byte()).view(2, -1)
            _data = pyg_nn.Data(x=data.x, edge_idnex=_edge_index)
            out.append(self.__gcn_nets[i](_data))

        return torch.cat(tuple(out), dim=1)
