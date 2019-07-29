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
                 cell_state_dim: int,
                 drug_state_dim: int,
                 cell_input_dim: Optional[int] = None,
                 drug_input_dim: Optional[int] = None,
                 cell_tower: Optional[nn.Module] = None,
                 drug_tower: Optional[nn.Module] = None,
                 dropout: float = 0.2):

        super(SimpleUno, self).__init__()

        self.__cell_tower = cell_tower if (cell_tower is not None) \
            else nn.Sequential(
                nn.Linear(cell_input_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(cell_state_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(cell_state_dim, cell_state_dim, bias=True),
                nn.BatchNorm1d(cell_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout))

        self.__drug_tower = drug_tower if (drug_tower is not None) \
            else nn.Sequential(
                nn.Linear(drug_input_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(drug_state_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(drug_state_dim, drug_state_dim, bias=True),
                nn.BatchNorm1d(drug_state_dim),
                nn.ReLU(),
                nn.Dropout(dropout))

        __inter_state_dim = int(state_dim ** 0.5)
        self.__pred_tower = nn.Sequential(
            nn.Linear(cell_state_dim + drug_state_dim + 1,
                      state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, __inter_state_dim, bias=True),
            nn.BatchNorm1d(__inter_state_dim),
            nn.ReLU(),

            nn.Linear(__inter_state_dim, 1, bias=True))

    def forward(self, cell_data, drug_data, dose):

        return self.__pred_tower(
            torch.cat((self.__cell_tower(cell_data),
                       self.__drug_tower(drug_data),
                       dose), dim=-1))
