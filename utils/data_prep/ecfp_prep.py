""" 
    File Name:          MoReL/ecfp_prep.py
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


def base64_ecfp_unpack(base64_ecfp: list) -> iter:

    ecfp = []
    for b64fp in base64_ecfp:

        bit_vect = ExplicitBitVect(0)
        ExplicitBitVect.FromBase64(bit_vect, b64fp)

        ecfp += [np.uint8(c) for c in bit_vect.ToBitString()]

    return np.array(ecfp, dtype=np.uint8).tolist()


def mol_to_base64_ecfp(mol: Chem.rdchem.Mol) -> list:

    return [AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=c.ECFP_N_BITS).ToBase64()
            for radius in c.ECFP_RADIUS]


def mol_to_ecfp(mol: Chem.rdchem.Mol) -> np.array:
    return base64_ecfp_unpack(mol_to_base64_ecfp(mol))


if __name__ == '__main__':

    smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)

    base64_ecfp1 = mol_to_base64_ecfp(mol1)
    print(base64_ecfp1)

    ecfp1 = base64_ecfp_unpack(base64_ecfp1)
    print(len(ecfp1))
    print(ecfp1)
