""" 
    File Name:          MoReL/mol_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/25/19
    Python Version:     3.5.4
    File Description:   

"""
import codecs
from rdkit import Chem

import utils.data_prep.config as c


def mol_to_str(mol: Chem.Mol) -> str:
    try:
        return codecs.encode(mol.ToBinary(), c.MOL_BINARY_ENCODING).decode()
    except:
        return None


def str_to_mol(mol_str: str) -> Chem.Mol:
    try:
        return Chem.Mol(codecs.decode(codecs.encode(mol_str),
                                      c.MOL_BINARY_ENCODING))
    except:
        return None
