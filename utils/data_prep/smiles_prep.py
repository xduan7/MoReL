""" 
    File Name:          MoReL/smiles_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

"""
import numpy as np
from rdkit import Chem
import utils.data_prep.config as c


def mol_to_smiles(mol: Chem.rdchem.Mol) -> str:

    return Chem.MolToSmiles(
        Mol=mol,
        allBoundsExplicit=c.ALL_BOUNDS_EXPLICIT,
        allHsExplicit=c.ALL_HS_EXPLICIT)


def tokenize_smiles(smiles: str,
                    token_dict: dict = c.TOKEN_DICT,
                    num_tokens: int = c.MAX_LEN_SMILES + 2) -> np.array:
    """
    This function takes a SMILES string and a list of common atoms and
    performs tokenization. Each Atom and bound will be tokenized into an
    integer number (np.uint8).

    No sanity check required.S

    :param smiles:
    :param token_dict:
    :param num_tokens: length of token array (including SOS, EOS, UNK, and PAD)

    :return:
    """

    return
