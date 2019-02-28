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
                 padding: bool = c.GRAPH_PADDING,
                 master_atom: bool = c.GRAPH_MASTER_ATOM) -> tuple:

    # Sanity check
    try:
        atom_list = mol.GetAtoms()
        num_atoms = len(atom_list)
        assert num_atoms <= c.MAX_NUM_ATOMS
    except:
        return None, None

    # num_atoms = mol.GetNumAtoms()
    ofs = 1 if master_atom else 0
    graph_dim = c.MAX_NUM_ATOMS + ofs if padding \
        else num_atoms + ofs

    # Feature for atoms/nodes
    nodes = np.zeros(
        shape=(graph_dim, len(c.ATOM_FEAT_FUNC_LIST)), dtype=np.int8)

    # Adjacency matrix, feature for bonds/edges
    edges = np.zeros(
        shape=(graph_dim, graph_dim, len(c.BOND_FEAT_FUNC_LIST) + 1),
        dtype=np.int8)

    # Iterate through all the atoms in the molecule
    for i, atom in enumerate(atom_list):
        for j, feat_func in enumerate(c.ATOM_FEAT_FUNC_LIST):
            nodes[i + ofs, j] = np.int8(feat_func(atom))

    # Iterate through all the bonds
    for i, j in list(itertools.product(
            range(num_atoms), range(num_atoms))):

        bond: Chem.Bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue

        # Indicator for connectivity, not actually a feature
        edges[i + ofs, j + ofs, 0] = np.int8(1)

        # Get all the bond features
        # Note that directional bond features will be inverted if necessary
        # TODO: test directional bonds
        for k, feat_func in enumerate(c.BOND_FEAT_FUNC_LIST):
            feat = feat_func(bond)
            if type(feat) in c.DIR_BOND_FEAT_TYPE_LIST and \
                    bond.GetBeginAtomIdx() != i:
                edges[i + ofs, j + ofs, k + 1] = -np.int8(feat)
            else:
                edges[i + ofs, j + ofs, k + 1] = np.int8(feat)

    # Note that the (virtual) master atom will connect to all the real atoms
    if master_atom:
        for i in range(num_atoms):
            edges[0, i + ofs, 0] = np.int8(1)
            edges[i + ofs, 0, 0] = np.int8(1)

    return nodes, edges


if __name__ == '__main__':

    smiles1 = 'C(=O)=O'
    # smiles1 = 'ClC(Cl)(F)F'
    # smiles1 = 'C(C1C(C(C(C(O1)O)O)O)O)O'
    # smiles1 = 'CCCCNC(=O)[C@@H]1CCCN(C(=O)CCC(C)C)C1'
    mol1 = Chem.MolFromSmiles(smiles1)
    n, e = mol_to_graph(mol1)
