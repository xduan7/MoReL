""" 
    File Name:          MoReL/reshape.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/25/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn


class Reshape(nn.Module):
    """
    Reshape layer for PyTorch.
    When shape is not given, this serves as a flatten layer.
    """
    def __init__(self, shape=None):
        super().__init__()
        self.__shape = shape

    def forward(self, input):
        return input.view(self.__shape)

