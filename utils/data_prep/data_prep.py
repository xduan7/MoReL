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
import os
import time
import json
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm.auto import tqdm

import utils.data_prep.config as c
from utils.data_prep.download import download



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


def inchi_to_mol_str(cid: int, inchi: str) -> tuple:
    try:
        mol: Chem.rdchem.Mol = Chem.MolFromInchi(inchi)
        assert mol is not None
        mol_str = mol_to_str(mol)
        return cid, mol, mol_str
    except AssertionError:
        return cid, None, None


def get_from_hdf5(cid: str, cid_grp: h5py.Group) -> np.array:
    return np.array(cid_grp.get(name=cid))


def feature_prep(pcba_only: bool,
                 count_atoms: bool,
                 compute_features: bool) -> None:
    download()

    # Prepare the PCBA CID set if necessary ###################################
    pcba_cid_set = set(pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1)) if pcba_only else set([])

    # Looping over all the CIDs set the molecule ##############################
    if compute_features and os.path.exists(c.CID_FEATURES_HDF5_PATH):
        msg = 'Feature HDF5 already exits. ' \
              'Press [Y] for overwrite, any other key to continue ... '
        if str(input(msg)).lower().strip() != 'y':
            return
    if (not compute_features) and os.path.exists(c.CID_MOL_STR_HDF5_PATH):
        msg = 'Molecule HDF5 already exits. ' \
              'Press [Y] for overwrite, any other key to continue ... '
        if str(input(msg)).lower().strip() != 'y':
            return

    print('Extracting molecule features ... ')

    # Book keeping for unused CIDs, atom occurrences, and number of CIDs
    unused_cid_list = []
    atom_dict = {}
    num_used_cid = 0

    if compute_features:
        # Features HDF5 and groups
        cid_features_hdf5 = h5py.File(c.CID_FEATURES_HDF5_PATH, 'w',
                                      libver='latest')
        cid_mol_str_hdf5_grp = \
            cid_features_hdf5.create_group(name='CID-Mol_str')
        cid_token_hdf5_grp = cid_features_hdf5.create_group(name='CID-token')
        cid_ecfp_hdf5_grp = cid_features_hdf5.create_group(name='CID-ECFP')
        cid_graph_hdf5_grp = cid_features_hdf5.create_group(name='CID-graph')
        cid_node_hdf5_grp = cid_graph_hdf5_grp.create_group(name='CID-node')
        cid_edge_hdf5_grp = cid_graph_hdf5_grp.create_group(name='CID-edge')
    else:
        cid_mol_str_hdf5 = h5py.File(c.CID_MOL_STR_HDF5_PATH, 'w',
                                     libver='latest')

    progress_bar = tqdm(
        total=len(pcba_cid_set),
        desc='extracting',
        ncols=160,
        miniters=1,
        bar_format='{desc}: {percentage:5.2f}%|'
                   '{bar}|'
                   '{n_fmt:>9}/{total_fmt:<9} '
                   '[Elapsed: {elapsed:<9}, '
                   'Estimated Remaining: {remaining:<9} '
                   '({rate_fmt:<10})]')

    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 15)):

        if compute_features:
            # Structure: (str(CID), Mol, str(Mol), tokens, ECFP, nodes, edges)
            chunk_cid_feature_list = Parallel(n_jobs=c.NUM_CORES)(
                delayed(inchi_to_features)(cid, row[1])
                for cid, row in chunk_cid_inchi_df.iterrows()
                if ((not c.PCBA_ONLY) or (cid in pcba_cid_set)))
        else:
            # Structure: (str(CID), str(Mol))
            chunk_cid_feature_list = Parallel(n_jobs=c.NUM_CORES)(
                delayed(inchi_to_mol_str)(cid, row[1])
                for cid, row in chunk_cid_inchi_df.iterrows()
                if ((not c.PCBA_ONLY) or (cid in pcba_cid_set)))

        # Loop over all the features extracted from this chunk, and:
        # * Check if the feature extraction is successful, if so, then
        # * Count the atoms if necessary
        # * Append the chunk lists to HDF5
        for cid_feature in chunk_cid_feature_list:

            if compute_features:
                cid, mol, mol_str, token, ecfp, node, edge = cid_feature
            else:
                cid, mol, mol_str = cid_feature

            # Sanity check
            if any(i is None for i in cid_feature):
                unused_cid_list.append(cid)
                continue
            else:
                num_used_cid += 1

            # Count the atoms in this molecule. Same atoms in a molecule only
            # count once. This information is used to give estimate of which
            # atoms should we tokenize
            if count_atoms:
                atom_set = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                for a in atom_set:
                    atom_dict[a] = (atom_dict[a] + 1) if a in atom_dict else 1

            # Write to HDF5
            # This is the actual bottleneck during data preparation. Feature
            # extraction is parallelized enough, but the writing part is
            # not. Moreover, if HDF5 libver is not set to 'latest',
            # the writing would be less than 1 MB per second later on in the
            # data preparation loop. This is because the internal structure
            # of HDF5, which is not efficient when we have too many dataset
            # inserted in a incremental fashion.
            # TODO: a possible solution to this is by writing in memory
            #  first and then construct such dataset
            if compute_features:
                cid_mol_str_hdf5_grp.create_dataset(
                    name=str(cid), data=mol_str)
                cid_token_hdf5_grp.create_dataset(name=str(cid), data=token)
                cid_ecfp_hdf5_grp.create_dataset(name=str(cid), data=ecfp)
                cid_node_hdf5_grp.create_dataset(name=str(cid), data=node)
                cid_edge_hdf5_grp.create_dataset(name=str(cid), data=edge)
            else:
                cid_mol_str_hdf5.create_dataset(name=str(cid), data=mol_str)

        # Update progress bar
        progress_bar.update(len(chunk_cid_feature_list))

    if compute_features:
        cid_features_hdf5.close()
    else:
        cid_mol_str_hdf5.close()

    # Saving metadata into files ##############################################
    # Convert atom count into frequency for further usage
    if count_atoms:
        for a in atom_dict:
            atom_dict[a] = atom_dict[a] / num_used_cid
        with open(c.ATOM_DICT_TXT_PATH, 'w') as f:
            json.dump(atom_dict, f, indent=4)

    # Dump unused CIDs for further usage
    with open(c.UNUSED_CID_TXT_PATH, 'w') as f:
        json.dump(unused_cid_list, f, indent=4)


def target_prep():

    # Return if target file already exists
    if os.path.exists(c.PCBA_CID_TARGET_DSCPTR_FILE_PATH):
        return
    print('Preparing dragon7 descriptor as target ...')

    # Load dragon7 descriptor targets #########################################
    pcba_cid_dscrptr_df = pd.read_csv(
        c.PCBA_CID_DSCPTR_FILE_PATH,
        sep='\t',
        header=0,
        index_col=0,
        na_values='na',
        dtype=str,
        usecols=['NAME', ] + c.TARGET_DSCRPTR_NAMES)

    # Please
    pcba_cid_dscrptr_df.index.names = ['CID']

    # Note that there are still 'na' values in this dataframe
    pcba_cid_dscrptr_df.to_csv(c.PCBA_CID_TARGET_DSCPTR_FILE_PATH)


def data_prep(pcba_only: bool,
              count_atoms: bool,
              compute_features: bool) -> None:
    download()
    feature_prep(pcba_only=pcba_only,
                 count_atoms=count_atoms,
                 compute_features=compute_features)
    target_prep()


if __name__ == '__main__':
    pass

    # data_prep(pcba_only=c.PCBA_ONLY,
    #           count_atoms=True,
    #           compute_features=False)
    # target_prep()
    #
    # # Processing atom dict and print out the atoms for tokenization
    # with open(c.ATOM_DICT_TXT_PATH, 'r') as f:
    #     tmp_atom_dict = json.load(f)
    #
    # # Construct and make sure that the atom dict
    # ordered_atoms = sorted(tmp_atom_dict,
    #                        key=tmp_atom_dict.__getitem__, reverse=True)
    # token_atoms = [a for a in ordered_atoms
    #                if tmp_atom_dict[a] > c.MIN_ATOM_FREQUENCY]
    # atom_token_dict = {a: Chem.Atom(a).GetAtomicNum() for a in token_atoms}
    # try:
    #     assert atom_token_dict == c.ATOM_TOKEN_DICT
    # except AssertionError:
    #     print('The atom dict extract from molecules differs '
    #           'from the one in MoReL/utils/data_prep/config.c')
    #
    # # Test the speed of mol_str -> features versus loading from hard drive
    # print('Test the speed of computing mol_str -> features '
    #       'versus loading features from disk ... ')
    # TEST_SIZE = 1024
    # with h5py.File(c.CID_FEATURES_HDF5_PATH, 'r') as f:
    #
    #     print('Opening HDF5 files and selecting the test indices ...')
    #     cid_mol_str_grp = f.get(name='CID-Mol_str')
    #     cid_token_grp = f.get(name='CID-token')
    #     cid_ecfp_grp = f.get(name='CID-ECFP')
    #     cid_graph_grp = f.get(name='CID-graph')
    #     cid_node_grp = cid_graph_grp.get(name='CID-node')
    #     cid_edge_grp = cid_graph_grp.get(name='CID-edge')
    #
    #     cid_list = list(cid_mol_str_grp.keys())
    #     test_cid_list = np.random.choice(
    #         cid_list, TEST_SIZE, replace=False)
    #
    #     print('Start testing ...')
    #     # Testing token (from SMILES)
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         # tmp_token = np.array(cid_token_grp.get(name=tmp_cid))
    #         tmp_token = get_from_hdf5(cid=tmp_cid, cid_grp=cid_token_grp)
    #         # print(tmp_token)
    #         # print(tmp_token.dtype)
    #     print("Get token from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol_str = str(get_from_hdf5(cid=tmp_cid,
    #                                         cid_grp=cid_mol_str_grp))
    #         tmp_mol = str_to_mol(tmp_mol_str)
    #         tmp_token = mol_to_token(tmp_mol)
    #         # print(tmp_token)
    #         # print(tmp_token.dtype)
    #     print("Get token from Mol:  \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     # Testing ECFP
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_ecfp = get_from_hdf5(cid=tmp_cid, cid_grp=cid_ecfp_grp)
    #         # tmp_ecfp = np.array(cid_ecfp_grp.get(name=tmp_cid))
    #         # print(tmp_ecfp)
    #     print("Get ECFP from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol_str = str(get_from_hdf5(cid=tmp_cid,
    #                                         cid_grp=cid_mol_str_grp))
    #         tmp_mol = str_to_mol(tmp_mol_str)
    #         tmp_ecfp = mol_to_ecfp(tmp_mol)
    #         # print(tmp_ecfp)
    #     print("Get ECFP from Mol:  \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     # Testing graphs
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_node = get_from_hdf5(cid=tmp_cid, cid_grp=cid_node_grp)
    #         tmp_edge = get_from_hdf5(cid=tmp_cid, cid_grp=cid_edge_grp)
    #         # tmp_node = cid_node_grp.get(name=tmp_cid)
    #         # tmp_edge = cid_edge_grp.get(name=tmp_cid)
    #         # print(tmp_node)
    #         # print(tmp_edge)
    #     print("Get graph from HDF5: \t%s seconds"
    #           % (time.time() - start_time))
    #
    #     start_time = time.time()
    #     for tmp_cid in test_cid_list:
    #         tmp_mol_str = str(get_from_hdf5(cid=tmp_cid,
    #                                         cid_grp=cid_mol_str_grp))
    #         tmp_mol = str_to_mol(tmp_mol_str)
    #         tmp_node, tmp_edge = mol_to_graph(tmp_mol)
    #         # print(tmp_node)
    #         # print(tmp_edge)
    #     print("Get graph from Mol:  \t%s seconds"
    #           % (time.time() - start_time))


def inchi_prep(pcba_only: bool) -> None:

    download()

    # Check if the CID-InChI HDF5 file already exist ##########################
    cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH \
        if pcba_only else c.PC_CID_INCHI_HDF5_PATH

    if os.path.exists(cid_inchi_hdf5_path):
        msg = 'CID-InChI HDF5 already exits. ' \
              'Press [Y] for overwrite, any other key to continue ... '
        if str(input(msg)).lower().strip() != 'y':
            return
        os.remove(cid_inchi_hdf5_path)

    cid_inchi_hdf5 = h5py.File(cid_inchi_hdf5_path, 'w', libver='latest')

    # Prepare the PCBA CID set if necessary ###################################
    pcba_cid_set = set(pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1)) if pcba_only else set([])

    # Looping over all the CIDs ###############################################
    def __check_inchi(cid, inchi):
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return cid, inchi
        else:
            return cid, None

    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 15)):

        chunk_cid_inchi_list = Parallel(n_jobs=c.NUM_CORES)(
            delayed(__check_inchi)(cid, row[1])
            for cid, row in chunk_cid_inchi_df.iterrows()
            if ((not pcba_only) or (cid in pcba_cid_set)))

        for cid, row in chunk_cid_inchi_list:
            if row is not None:
                cid_inchi_hdf5.create_dataset(name=str(cid), data=row[1])

