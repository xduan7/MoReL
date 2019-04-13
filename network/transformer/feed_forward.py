""" 
    File Name:          MoReL/feed_forward.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/7/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 mid_dim: int = 1024,
                 dropout: float = 0.0):

        super().__init__()

        self.__module = nn.Sequential(
            nn.Linear(emb_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, emb_dim))

    def forward(self, x):
        return self.__module(x)
