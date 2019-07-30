""" 
    File Name:          MoReL/simple_uno.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               7/28/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
from torch import nn
from typing import Optional


# Simple Uno-like model
class SimpleUno(nn.Module):

    def __init__(self,
                 state_dim: int,
                 dose_info: bool,

                 cell_state_dim: int,
                 drug_state_dim: int,

                 cell_input_dim: Optional[int] = None,
                 drug_input_dim: Optional[int] = None,
                 cell_tower: Optional[nn.Module] = None,
                 drug_tower: Optional[nn.Module] = None,
                 dropout_rate: float = 0.2,
                 sigmoid_output: bool = True):

        super(SimpleUno, self).__init__()
        self.__dose_info = dose_info
        self.__sigmoid_output = sigmoid_output

        self.__cell_tower = cell_tower if (cell_tower is not None) \
            else nn.Sequential(
                nn.Linear(cell_input_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU(),

                nn.Linear(cell_state_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU(),

                nn.Linear(cell_state_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU())

        self.__drug_tower = drug_tower if (drug_tower is not None) \
            else nn.Sequential(
                nn.Linear(drug_input_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU(),

                nn.Linear(drug_state_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU(),

                nn.Linear(drug_state_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU())

        __inter_state_dim = int(state_dim ** 0.5)
        self.__pred_tower = nn.Sequential(
            nn.Linear((cell_state_dim + drug_state_dim + 1) if dose_info
                      else (cell_state_dim + drug_state_dim),
                      state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(state_dim, __inter_state_dim, bias=True),
            nn.BatchNorm1d(__inter_state_dim),
            nn.ReLU(),

            nn.Linear(__inter_state_dim, 1, bias=True))

    def forward(self, cell_data, drug_data, dose=None):

        __cell_latent_vec = self.__cell_tower(cell_data)
        __drug_latent_vec = self.__drug_tower(drug_data)

        __latent_vec = (__cell_latent_vec, __drug_latent_vec, dose) \
            if self.__dose_info else (__cell_latent_vec, __drug_latent_vec)
        __pred = self.__pred_tower(torch.cat(__latent_vec, dim=-1))

        return torch.sigmoid(__pred) if self.__sigmoid_output else __pred
