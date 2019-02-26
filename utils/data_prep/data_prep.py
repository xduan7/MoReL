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
from rdkit import Chem
from rdkit import RDLogger

import utils.data_prep.config as c
from utils.data_prep.download import download
from utils.data_prep.smiles_prep import mol_to_token, mol_to_smiles
from utils.data_prep.mol_prep import mol_to_str, str_to_mol
from utils.data_prep.ecfp_prep import mol_to_ecfp
from utils.data_prep.graph_prep import mol_to_graph


def data_prep(pcba_only=True):

    # Suppress unnecessary RDkit warnings and errors
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    # Download and prepare the PCBA CID set if necessary ######################
    download()
    pcba_cid_set = set(pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1)) if pcba_only else set([])

    # Looping over all the CIDs set the molecule ##############################
    # Book keeping for unused CIDs, atom occurrences, and number of CIDs
    unused_cid_set = set([])
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

    # TODO: a progress indicator here would be very nice.
    # TODO: also suppress the WARNINGs and ERRORs from RDkit
    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 12)):

        chunk_cid_feature_list = []
        for cid, row in chunk_cid_inchi_df.iterrows():

            # Skip this compound if it is not in PCBA and the dataset is PCBA
            if (cid not in pcba_cid_set) and c.PCBA_ONLY:
                continue

            mol: Chem.rdchem.Mol = Chem.MolFromInchi(row[1])
            if mol is None:
                unused_cid_set.add(cid)
                continue

            # Get the SMILES strings and molecule representation strings
            # Note that mol from smiles from mol will keep mol and smiles
            # consistent, which is important in tokenization
            smiles: str = mol_to_smiles(mol)
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                unused_cid_set.add(cid)
                continue

            mol_str: str = mol_to_str(mol)

            if mol.GetNumAtoms() > c.MAX_NUM_ATOMS or \
                    len(smiles) > c.MAX_LEN_SMILES:
                unused_cid_set.add(cid)
                continue

            # Note that all the featurization parameters are in config.py
            token = mol_to_token(mol)
            # ecfp = mol_to_ecfp(mol)
            # graph = mol_to_graph(mol)

            # Count the atoms in this molecule
            # Same atoms in a molecule only count once
            atom_set = set([a.GetSymbol() for a in mol.GetAtoms()])
            for a in atom_set:
                atom_dict[a] = (atom_dict[a] + 1) if a in atom_dict else 1

            # Append extracted data to list and prepare for storage
            # chunk_cid_feature_list.append(
            #     (str(cid), (mol_str, token, ecfp, graph)))
            chunk_cid_feature_list.append(cid)

        # Append the chunk lists to HDF5
        # for cid, (mol_str, token, ecfp, (node, edge)) in \
        #         chunk_cid_feature_list:
        #     cid_mol_str_hdf5_grp.create_dataset(name=cid, data=mol_str)
        #     cid_token_hdf5_grp.create_dataset(name=cid, data=token)
        #     cid_ecfp_hdf5_grp.create_dataset(name=cid, data=ecfp)
        #     cid_node_hdf5_grp.create_dataset(name=cid, data=node)
        #     cid_edge_hdf5_grp.create_dataset(name=cid, data=edge)

        num_used_cid += len(chunk_cid_feature_list)

    cid_features_hdf5.close()

    # Saving metadata into files ##############################################
    # Convert atom count into frequency for further usage
    for a in atom_dict:
        atom_dict[a] = atom_dict[a] / num_used_cid
    with open(c.ATOM_DICT_TXT_PATH, 'w') as f:
        json.dump(atom_dict, f, indent=4)

    # Dump unused CIDs for further usage
    with open(c.UNUSED_CID_TXT_PATH, 'w') as f:
        json.dump(unused_cid_set, f, indent=4)

    return pcba_cid_set


if __name__ == '__main__':

    data_prep()

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
