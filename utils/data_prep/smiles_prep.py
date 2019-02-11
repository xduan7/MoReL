""" 
    File Name:          MoReL/smiles_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

"""
from rdkit import Chem


def mol_to_smiles(mol: Chem.rdchem.Mol):

    # Get the canonical SMILES string from a molecule
    smiles_ = Chem.MolToSmiles(mol, allBondsExplicit=True)
    smiles = Chem.MolToSmiles(mol, allBondsExplicit=True)

    print("SMILES canonical: %s", smiles)
    print("SMILES with explicit bounds: %s", smiles_)




    return


def tokenize_simles(smiles: str,
                    ):
    return




