""" 
    File Name:          MoReL/parameter_counting.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/22/19
    Python Version:     3.5.4
    File Description:   

"""
import torch.nn as nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
