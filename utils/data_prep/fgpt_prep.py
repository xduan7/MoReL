""" 
    File Name:          MoReL/fgpt_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        TODO: extend this module into different fingerprint types
"""
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import utils.data_prep.config as c


# def fgpt_to_base64(fgpt: ExplicitBitVect) -> str:
#     return fgpt.ToBase64()


def base64_to_fgpt(base64_str: str) -> np.array:

    bit_vect = ExplicitBitVect(0)
    ExplicitBitVect.FromBase64(bit_vect, base64_str)

    fgpt = [np.uint8(c) for c in bit_vect.ToBitString()]
    return np.array(fgpt, dtype=np.uint8)


def mol_to_fgpt(mol: Chem.rdchem.Mol):

    for radius in c.FGPT_RADIUS:
        fp_bit_str: str = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=c.FGPT_N_BITS).ToBitString()

    return


if __name__ == '__main__':

    import sys

    smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)

    # bv: ExplicitBitVect = AllChem.GetMorganFingerprintAsBitVect(
    #     mol1, radius=2, nBits=4096)
    #
    # fp_bin = AllChem.GetMorganFingerprintAsBitVect(
    #     mol1, radius=2, nBits=4096).ToBinary()
    #
    # fp_bin_str = AllChem.GetMorganFingerprintAsBitVect(
    #     mol1, radius=2, nBits=4096).ToBitString()
    #
    # print(fp_bin_str)

    fp_b64_str = AllChem.GetMorganFingerprintAsBitVect(
        mol1, radius=2, nBits=4096).ToBase64()

    print(base64_to_fgpt(fp_b64_str))

    # codecs.encode(mol.ToBinary(), c.MOL_BINARY_ENCODING).decode()

    # p = pickle.dumps(fp_bin)
    # print(sys.getsizeof(p))
    #
    # p = pickle.dumps(fp_bin_str)
    # print(sys.getsizeof(p))

    p = pickle.dumps(fp_b64_str)
    print(sys.getsizeof(p))
