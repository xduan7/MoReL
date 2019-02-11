""" 
    File Name:          MoReL/graph_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/5/19
    Python Version:     3.5.4
    File Description:   
        This implementation is based on:
            https://github.com/HIPS/neural-fingerprint/blob/master/\
                neuralfingerprint/graph_prep.py
        which is the git repo for https://arxiv.org/pdf/1509.09292.pdf

        And
            https://github.com/deepchem/deepchem/blob/master/\
                deepchem/feat/graph_features.py
        which is the git repo for DeepChem
"""
import numpy as np
from rdkit import Chem


def mol_to_graph(mol: Chem.rdchem.Mol):
    return


def annotate_graph(atoms: iter, adj_mat: np.matrix):
    return


