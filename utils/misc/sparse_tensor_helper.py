""" 
    File Name:          MoReL/sparse_tensor_helper.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/8/19
    Python Version:     3.5.4
    File Description:   

"""
import torch


def is_sparse(tensor: torch.Tensor):
    return tensor.type().startswith('torch.sparse')


def is_dense(tensor: torch.Tensor):
    return not tensor.type().startswith('torch.sparse')


def to_dense(tensor: torch.Tensor):
    return tensor if is_dense(tensor) else tensor.to_dense()


def to_sparse(tensor: torch.Tensor):
    return tensor if is_sparse(tensor) else tensor.to_sparse()
