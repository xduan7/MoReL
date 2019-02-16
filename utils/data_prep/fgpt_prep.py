""" 
    File Name:          MoReL/fgpt_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        TODO: extend this module into different fingerprint types
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import utils.data_prep.config as c


# There are different ways to encode fingerprint
# * np.array of np.uint8
# * base64 encoded ExplicitBitVect
# * sparse matrix


def base64_fgpt_unpack(base64_fgpt: list) -> np.array:

    fgpt = []
    for b64fp in base64_fgpt:

        bit_vect = ExplicitBitVect(0)
        ExplicitBitVect.FromBase64(bit_vect, b64fp)

        fgpt += [np.uint8(c) for c in bit_vect.ToBitString()]

    return np.array(fgpt, dtype=np.uint8).reshape(-1)


def mol_to_base64_fgpt(mol: Chem.rdchem.Mol) -> list:

    return [AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=c.FGPT_N_BITS).ToBase64()
            for radius in c.FGPT_RADIUS]


if __name__ == '__main__':

    smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)

    base64_fgpt1 = mol_to_base64_fgpt(mol1)
    print(base64_fgpt1)

    fgpt1 = base64_fgpt_unpack(base64_fgpt1)
    print(len(fgpt1))
    print(fgpt1)
