""" 
    File Name:          MoReL/ecfp_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        TODO: extend this module into different fingerprint types
"""
from rdkit import Chem
from rdkit.Chem import AllChem


def mol_to_ecfp(mol: Chem.rdchem.Mol):
    return


if __name__ == '__main__':

    import sys

    smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)

    bv1 = AllChem.GetMorganFingerprint(mol1, radius=2, nBits=1024)
    print(bv1)
    print(sys.getsizeof(bv1))
