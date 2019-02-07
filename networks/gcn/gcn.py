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

    def __init__(self, nfeat, nhid, nclass, dropout):

        super().__init__()

        self.gc1 = GraphConvLayer(nfeat, nhid)
        self.gc2 = GraphConvLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

