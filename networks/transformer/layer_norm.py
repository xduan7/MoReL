""" 
    File Name:          MoReL/layer_norm.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self,
                 emb_dim: int,
                 epsilon: float = 1e-6):

        super().__init__()

        self.__emb_dim = emb_dim

        # Adaptive gain and bias for learning
        self.__gain = nn.Parameter(torch.ones(self.__emb_dim))
        self.__bias = nn.Parameter(torch.zeros(self.__emb_dim))

        self.__epsilon = epsilon

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.std(-1, keepdim=True)

        return self.__gain * (x - mu) / (sigma + self.__epsilon) + self.__bias
