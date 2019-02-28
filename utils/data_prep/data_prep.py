""" 
    File Name:          MoReL/data_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        Prepare training/validation data.

        This file is pretty much the summation of mol_prep, smiles_prep,
        ecfp_prep, and graph_prep.
"""
import time
import json
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm

import utils.data_prep.config as c
from utils.data_prep.download import download
from utils.data_prep.smiles_prep import mol_to_token
from utils.data_prep.mol_prep import mol_to_str
from utils.data_prep.ecfp_prep import mol_to_ecfp
from utils.data_prep.graph_prep import mol_to_graph


def mol_to_features(cid: int, mol: Chem.Mol) -> tuple:
    try:
        assert mol
        mol_str = mol_to_str(mol)
        assert mol_str is not None
        token = mol_to_token(mol)
        assert token is not None
        ecfp = mol_to_ecfp(mol)
        assert ecfp is not None
        node, edge = mol_to_graph(mol)
        assert node is not None
        assert edge is not None
        return cid, mol, mol_str, token, ecfp, node, edge
    except AssertionError:
        return cid, None, None, None, None, None, None


def inchi_to_features(cid: int, inchi: str) -> tuple:
    try:
        mol: Chem.rdchem.Mol = Chem.MolFromInchi(inchi)
        assert mol is not None
        return mol_to_features(cid, mol)
    except AssertionError:
        return cid, None, None, None, None, None, None


def data_prep(pcba_only=True,
              counting_atoms=False):

    # # Suppress unnecessary RDkit warnings and errors
    # RDLogger.logger().setLevel(RDLogger.CRITICAL)

    # Download and prepare the PCBA CID set if necessary ######################
    download()
    pcba_cid_set = set(pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1)) if pcba_only else set([])

    # Looping over all the CIDs set the molecule ##############################
    print('Featurizing molecules ... ')

    # Book keeping for unused CIDs, atom occurrences, and number of CIDs
    unused_cid_list = []
    atom_dict = {}
    num_used_cid = 0

    # HDF5 and groups
    cid_features_hdf5 = h5py.File(c.CID_FEATURES_HDF5_PATH, 'w')
    cid_mol_str_hdf5_grp = cid_features_hdf5.create_group(name='CID-Mol_str')
    cid_token_hdf5_grp = cid_features_hdf5.create_group(name='CID-token')
    cid_ecfp_hdf5_grp = cid_features_hdf5.create_group(name='CID-ECFP')
    cid_graph_hdf5_grp = cid_features_hdf5.create_group(name='CID-graph')
    cid_node_hdf5_grp = cid_graph_hdf5_grp.create_group(name='CID-node')
    cid_edge_hdf5_grp = cid_graph_hdf5_grp.create_group(name='CID-edge')

    # TODO: probably some ways to parallelize the inner loop?
    progress_bar = tqdm(
        total=len(pcba_cid_set),
        desc='featurization',
        ncols=120,
        miniters=1,
        bar_format='{desc}: {percentage:02.2f}%|'
                   '{bar}|'
                   '{n_fmt:>10}/{total_fmt:<10} '
                   '[Elapsed: {elapsed:<10}, Remaining: {remaining:<10} '
                   '({rate_fmt:<10})]')

    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 15)):

        # chunk_cid_feature_list = []
        # for cid, row in chunk_cid_inchi_df.iterrows():
        #
        #     # Skip this compound if it is not in PCBA and the dataset is PCBA
        #     if (cid not in pcba_cid_set) and c.PCBA_ONLY:
        #         continue
        #     chunk_num_cid += 1
        #
        #     mol: Chem.rdchem.Mol = Chem.MolFromInchi(row[1])
        #
        #     # Get the SMILES strings and molecule representation strings
        #     mol_str: str = mol_to_str(mol)
        #
        #     # Note that all the featurization parameters are in config.py
        #     tokens = mol_to_token(mol)
        #     ecfp = mol_to_ecfp(mol)
        #     nodes, edges = mol_to_graph(mol)
        #
        #     # Gather all the features and validate all of them
        #     # Note that dimension restrictions are enforced in each
        #     # featurization functions independently
        #     features = (mol_str, tokens, ecfp, nodes, edges)
        #
        #     # Count the atoms in this molecule
        #     # Same atoms in a molecule only count once
        #     # atom_set = set([a.GetSymbol() for a in mol.GetAtoms()])
        #     # for a in atom_set:
        #     #     atom_dict[a] = (atom_dict[a] + 1) if a in atom_dict else 1
        #
        #     # Append extracted data to list and prepare for storage
        #     chunk_cid_feature_list.append(
        #         (str(cid), (mol_str, tokens, ecfp, nodes, edges)))
        #     chunk_cid_feature_list.append(cid)

        # Structure: (str(CID), Mol, str(Mol), tokens, ECFP, nodes, edges)
        chunk_cid_feature_list = Parallel(n_jobs=c.NUM_CORES)(
            delayed(inchi_to_features)(cid, row[1])
            for cid, row in chunk_cid_inchi_df.iterrows()
            if ((not c.PCBA_ONLY) or (cid in pcba_cid_set)))

        # Loop over all the features extracted from this chunk, and:
        # * Check if the feature extraction is successful, if so, then
        # * Count the atoms if necessary
        # * Append the chunk lists to HDF5
        for cid_feature in chunk_cid_feature_list:
            cid, mol, mol_str, token, ecfp, node, edge = cid_feature

            # Sanity check
            if any(i is None for i in cid_feature):
                unused_cid_list.append(cid)
                continue
            else:
                num_used_cid += 1

            # Count the atoms in this molecule. Same atoms in a molecule only
            # count once. This information is used to give estimate of which
            # atoms should we tokenize
            if counting_atoms:
                atom_set = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                for a in atom_set:
                    atom_dict[a] = (atom_dict[a] + 1) if a in atom_dict else 1

            # Write to HDF5
            cid_mol_str_hdf5_grp.create_dataset(name=str(cid), data=mol_str)
            cid_token_hdf5_grp.create_dataset(name=str(cid), data=token)
            cid_ecfp_hdf5_grp.create_dataset(name=str(cid), data=ecfp)
            cid_node_hdf5_grp.create_dataset(name=str(cid), data=node)
            cid_edge_hdf5_grp.create_dataset(name=str(cid), data=edge)

        # Update progress bar
        progress_bar.update(len(chunk_cid_feature_list))

    cid_features_hdf5.close()

    # Saving metadata into files ##############################################
    # Convert atom count into frequency for further usage
    if counting_atoms:
        for a in atom_dict:
            atom_dict[a] = atom_dict[a] / num_used_cid
        with open(c.ATOM_DICT_TXT_PATH, 'w') as f:
            json.dump(atom_dict, f, indent=4)

    # Dump unused CIDs for further usage
    with open(c.UNUSED_CID_TXT_PATH, 'w') as f:
        json.dump(unused_cid_list, f, indent=4)

    return pcba_cid_set


if __name__ == '__main__':

    data_prep(counting_atoms=True)

    # Processing atom dict and print out the atoms for tokenization
    with open(c.ATOM_DICT_TXT_PATH, 'r') as f:
        tmp_atom_dict = json.load(f)

    # Construct and make sure that the atom dict
    ordered_atoms = sorted(tmp_atom_dict,
                           key=tmp_atom_dict.__getitem__, reverse=True)
    token_atoms = [a for a in ordered_atoms
                   if tmp_atom_dict[a] > c.MIN_ATOM_FREQUENCY]
    atom_token_dict = {a: Chem.Atom(a).GetAtomicNum() for a in token_atoms}
    try:
        assert atom_token_dict == c.ATOM_TOKEN_DICT
    except AssertionError:
        print('The atom dict extract from molecules differs '
              'from the one in MoReL/utils/data_prep/config.c')

    # Test out the speed of mol_str -> features versus loading from hard drive
    # TEST_SIZE = 1
    # with h5py.File(c.CID_FEATURES_HDF5_PATH, 'r') as f:
    #
    #     cid_mol_str_grp = f.get(name='CID-Mol_str')
    #     cid_token_grp = f.get(name='CID-token')
    #     cid_ecfp_grp = f.get(name='CID-ECFP')
    #     cid_graph_grp = f.get(name='CID-graph')
    #     cid_node_grp = cid_graph_grp.get(name='CID-node')
    #     cid_edge_grp = cid_graph_grp.get(name='CID-edge')
    #
    #     cid_list = list(cid_mol_str_grp.keys())
    #     test_cid_list = np.random.choice(
    #         cid_list, TEST_SIZE, replace=False).astype(str)
    #
    #     # Testing token (from SMILES)
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_token = cid_token_grp.get(name=tmp_cid)
    #         print(tmp_token)
    #     print("Get token from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol = str_to_mol(cid_mol_str_grp.get(name=tmp_cid))
    #         tmp_token = mol_to_token(tmp_mol)
    #         print(tmp_token)
    #     print("Get token from Mol: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     # Testing ECFP
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_ecfp = cid_ecfp_grp.get(name=tmp_cid)
    #         print(tmp_ecfp)
    #     print("Get ECFP from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol = str_to_mol(cid_mol_str_grp.get(name=tmp_cid))
    #         tmp_ecfp = mol_to_ecfp(tmp_mol)
    #         print(tmp_ecfp)
    #     print("Get ECFP from Mol: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     # Testing graphs
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_node = cid_node_grp.get(name=tmp_cid)
    #         tmp_edge = cid_edge_grp.get(name=tmp_cid)
    #         print(tmp_node)
    #         print(tmp_edge)
    #     print("Get graph from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol = str_to_mol(cid_mol_str_grp.get(name=tmp_cid))
    #         tmp_node, tmp_edge = mol_to_graph(tmp_mol)
    #         print(tmp_node)
    #         print(tmp_edge)
    #     print("Get ECFP from Mol: \t%s seconds"
    #           % (time.time() - start_time))
