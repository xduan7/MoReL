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
import itertools

import numpy as np
from rdkit import Chem

# mol_to_graph(sparse=? dense? padding?)
# sparse_graph_to_dense()
# pad_graph(dense graph as input)
import utils.data_prep.config as c


def mol_to_graph(mol: Chem.rdchem.Mol,
                 padding: bool = True,
                 master_atom: bool = True) -> tuple:

    num_atoms = mol.GetNumAtoms()
    ofs = 1 if master_atom else 0
    num_annotation = c.MAX_NUM_ATOMS + ofs if padding \
        else num_atoms + ofs

    # Annotation is the feature for atoms/nodes
    annotation = np.zeros(
        shape=(num_annotation, len(c.ATOM_FEAT_FUNC_LIST)), dtype=np.int8)

    # Adjacency matrix is the feature for bonds/edges
    adj_matrix = np.zeros(
        shape=(num_annotation, num_annotation,
               len(c.BOND_FEAT_FUNC_LIST) + ofs),
        dtype=np.int8)

    # Iterate through all the atoms in the molecule
    for i, atom in enumerate(mol.GetAtoms()):
        for j, feat_func in enumerate(c.ATOM_FEAT_FUNC_LIST):
            annotation[i + ofs, j] = np.int8(feat_func(atom))

    # Iterate through all the bonds
    for i, j in list(itertools.product(range(num_atoms), range(num_atoms))):

        bond: Chem.Bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue

        for k, feat_func in enumerate(c.BOND_FEAT_FUNC_LIST):

            feat = feat_func(bond)

            if type(feat) in c.DIR_BOND_FEAT_TYPE_LIST and \
                    bond.GetBeginAtomIdx() != i:
                adj_matrix[i + ofs, j + ofs, k + ofs] = -np.int8(feat)
            else:
                adj_matrix[i + ofs, j + ofs, k + ofs] = np.int8(feat)

    # Note that the (virtual) master atom will connect to all the real atoms
    if master_atom:
        for i in range(num_atoms):
            adj_matrix[0, i + ofs, 0] = np.int8(1)
            adj_matrix[i + ofs, 0, 0] = np.int8(1)

    return annotation, adj_matrix


def annotate_graph(atoms: iter, adj_mat: np.matrix):
    return


if __name__ == '__main__':
    smiles1 = 'ClC(Cl)(F)F'
    # smiles1 = 'C(C1C(C(C(C(O1)O)O)O)O)O'
    # smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)

    nodes, edges = mol_to_graph(mol1, padding=True)
