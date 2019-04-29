""" 
    File Name:          MoReL/residue_block.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/29/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseResBlock(nn.Module):

    def __init__(self,
                 state_dim: int,
                 num_layers: int,
                 dropout: float):

        super(DenseResBlock, self).__init__()
        self.__dropout = dropout

        self.__layers = nn.ModuleList(
            [nn.Linear(state_dim, state_dim) for _ in range(num_layers)])

    def forward(self, x, dropout=None):

        # This part is actually for UQ, where we need to tweak the dropout
        # in order to quantify the uncertainty.
        if dropout is None:
            p = self.__dropout if self.training else 0.
        else:
            p = dropout

        residual = x
        for layer in self.__layers:
            x = F.dropout(F.relu(layer(x)), p=p)

        return x + residual
