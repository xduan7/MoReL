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
import json
import codecs

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

import utils.data_prep.config as c
from utils.data_prep.download import download
from utils.data_prep.ecfp_prep import mol_to_base64_ecfp, mol_to_ecfp, \
    base64_ecfp_unpack
from utils.data_prep.graph_prep import mol_to_graph
from utils.data_prep.smiles_prep import mol_to_smiles, tokenize_smiles


def mol_to_str(mol: Chem.Mol):
    return codecs.encode(mol.ToBinary(), c.MOL_BINARY_ENCODING).decode()


def str_to_mol(mol_str: str):
    try:
        return Chem.Mol(codecs.decode(codecs.encode(mol_str),
                                      c.MOL_BINARY_ENCODING))
    except:
        return None


def data_prep(pcba_only=True):

    # Download and unpack data into raw data fodler ###########################
    download()

    # Split training and validation data ######################################
    pcba_cid_array = pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1).astype(np.int32)

    trn_cid_array, val_cid_array = train_test_split(
        pcba_cid_array,
        test_size=c.VALIDATION_SIZE,
        random_state=c.RANDOM_STATE)

    # Transforming all the array into set for faster lookup
    # l = list(range(4000000))
    # a = np.array(l)
    # s = set(l)
    # %timeit random.randint(0, 3999999) in l
    # 19.4 ms ± 3.32 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit random.randint(0, 3999999) in a
    # 3.04 ms ± 55.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit random.randint(0, 3999999) in s
    # 1.33 µs ± 7.42 ns per loop \
    # (mean ± std. dev. of 7 runs, 1000000 loops each)

    pcba_cid_set = set(pcba_cid_array)

    # Note that: trn_cid_set + val_cid_set == pcba_cid_set
    trn_cid_set = set(trn_cid_array)
    val_cid_set = set(val_cid_array)

    # Looping over all the CIDs set the molecule ##############################

    # Book keeping for unused CIDs, atom occurrences, and number of rows
    unused_cid = []
    atom_dict = {}
    num_cid_mol = 0

    # HDF5 file closed after looping
    # cid_mol_hdf5 = pd.HDFStore(c.CID_MOL_HDF5_PATH, mode='w')
    # cid_smiles_hdf5 = pd.HDFStore(c.CID_SMILES_HDF5_PATH, mode='w')
    # cid_tokenized_smiles_hdf5 = pd.HDFStore(c.CID_TOKENIZED_SMILES_HDF5_PATH,
    #                                         mode='w')
    # cid_base64_ecfp_hdf5 = pd.HDFStore(c.CID_BASE64_ECFP_HDF5_PATH, mode='w')
    # cid_ecfp_hdf5 = pd.HDFStore(c.CID_ECFP_HDF5_PATH, mode='w')
    # cid_graph_hdf5 = pd.HDFStore(c.CID_GRAPH_HDF5_PATH, mode='w')

    # Temp strcuture for all the CIDs and SMILES
    cid_smiles_list = []

    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 12)):

        chunk_cid_mol_list = []
        chunk_cid_smiles_list = []
        chunk_cid_tokenized_smiles_list = []
        chunk_cid_base64_ecfp_list = []
        chunk_cid_ecfp_list = []
        chunk_cid_graph_list = []

        for cid, row in chunk_cid_inchi_df.iterrows():

            # Skip this compound if it is not in PCBA and the dataset is PCBA
            # Note that
            if cid not in pcba_cid_set and c.PCBA_ONLY:
                continue

            mol: Chem.rdchem.Mol = Chem.MolFromInchi(row[1])
            if mol is None:
                unused_cid.append(cid)
                continue

            # Get the SMILES strings and molecule representation strings
            smiles: str = mol_to_smiles(mol)
            mol_str: str = mol_to_str(mol)

            # Skip the molecules with too many atoms or lengthy SMILES
            if mol.GetNumAtoms() > c.MAX_NUM_ATOMS or \
                    len(smiles) > c.MAX_LEN_SMILES or \
                    len(mol_str) > c.MAX_LEN_MOL_STR:
                unused_cid.append(cid)
                continue

            # Note that all the featurization parameters are in config.py
            # tokenized_smiles= tokenize_smiles(smiles)
            # base64_ecfp = mol_to_base64_ecfp(mol)
            # ecfp = base64_ecfp_unpack(base64_ecfp)
            # graph = mol_to_graph(mol)

            # Count the atoms in this molecule
            # Same atoms in a molecule only count once
            atom_set = set([a.GetSymbol() for a in mol.GetAtoms()])
            for a in atom_set:
                atom_dict[a] = (atom_dict[a] + 1) if a in atom_dict else 1

            # Append extracted data to list and prepare for storage
            cid_smiles_list.append([cid, smiles])
            # chunk_cid_mol_list.append([cid, mol_str])
            # chunk_cid_smiles_list.append([cid, smiles])
            # chunk_cid_tokenized_smiles_list.append([cid, tokenized_smiles])
            # chunk_cid_base64_ecfp_list.append([cid, base64_ecfp])
            # chunk_cid_ecfp_list.append([cid, ecfp])
            # chunk_cid_graph_list.append([cid, graph])

        # Append the chunk lists to HDF5
        # Molecules
        # chunk_cid_mol_df = pd.DataFrame(chunk_cid_mol_list,
        #                                 columns=['CID', 'Mol'])
        # chunk_cid_mol_df.set_index('CID', inplace=True)
        # cid_mol_hdf5.append(key='CID-Mol',
        #                     value=chunk_cid_mol_df,
        #                     min_itemsize=c.MAX_LEN_MOL_STR)

        # SMILES strings
        # chunk_cid_smiles_df = pd.DataFrame(chunk_cid_smiles_list,
        #                                    columns=['CID', 'SMILES'])
        # chunk_cid_smiles_df.set_index('CID', inplace=True)
        # cid_smiles_hdf5.append(key='CID-SMILES',
        #                        value=chunk_cid_smiles_df,
        #                        min_itemsize=c.MAX_LEN_SMILES)

        # Tokenized SMILES
        # chunk_cid_tokenized_smiles_df = \
        #     pd.DataFrame(chunk_cid_tokenized_smiles_list,
        #                  columns=['CID', 'tokenized_SMILES'])
        # chunk_cid_tokenized_smiles_df.set_index('CID', inplace=True)
        # cid_tokenized_smiles_hdf5.append(
        #     key='CID-tokenized_SMILES',
        #     value=chunk_cid_tokenized_smiles_df,
        #     min_itemsize=c.MAX_LEN_TOKENIZED_SMILES)

        # Base64 ECFP
        # chunk_cid_base64_ecfp_df = \
        #     pd.DataFrame(chunk_cid_base64_ecfp_list,
        #                  columns=['CID', 'base64_ECFP'])
        # chunk_cid_base64_ecfp_df.set_index('CID', inplace=True)
        # cid_base64_ecfp_hdf5.append(key='CID-base64_ECFP',
        #                             value=chunk_cid_base64_ecfp_df)

        # ECFP
        # chunk_cid_ecfp_df = pd.DataFrame(chunk_cid_ecfp_list,
        #                                  columns=['CID', 'ECFP'])
        # chunk_cid_ecfp_df.set_index('CID', inplace=True)
        # cid_ecfp_hdf5.append(key='CID-ECFP',
        #                      value=chunk_cid_ecfp_df)

        # Graph
        # chunk_cid_grahp_df = pd.DataFrame(chunk_cid_graph_list,
        #                                   columns=['CID', 'graph'])
        # chunk_cid_grahp_df.set_index('CID', inplace=True)
        # cid_graph_hdf5.append(key='CID-graph',
        #                       value=chunk_cid_grahp_df)

        # num_cid_mol += len(chunk_cid_mol_list)

    # cid_mol_hdf5.close()
    # cid_smiles_hdf5.close()
    # cid_tokenized_smiles_hdf5.close()
    # cid_base64_ecfp_hdf5.close()
    # cid_ecfp_hdf5.close()
    # cid_graph_hdf5.close()

    # Saving metadata into files ##############################################
    cid_smiles_df = pd.DataFrame(cid_smiles_list,
                                 columns=['CID', 'SMILES'])
    cid_smiles_df.set_index('CID', inplace=True)
    cid_smiles_df.to_csv(c.CID_SMILES_CSV_PATH)

    # Convert atom count into frequency for further usage
    # for a in atom_dict:
    #     atom_dict[a] = atom_dict[a] / num_cid_mol
    # with open(c.ATOM_DICT_TXT_PATH, 'w') as f:
    #     json.dump(atom_dict, f, indent=4)
    #
    # # Dump unused CIDs for further usage
    # with open(c.UNUSED_CID_TXT_PATH, 'w') as f:
    #     json.dump(unused_cid, f, indent=4)

    return


if __name__ == '__main__':

    data_prep()

    # Processing atom dict and print out the atoms for tokenization
    with open(c.ATOM_DICT_TXT_PATH, 'r') as f:
        atom_dict = json.load(f)

    ordered_atoms = sorted(atom_dict, key=atom_dict.__getitem__, reverse=True)

    token_atoms = [a for a in ordered_atoms
                   if atom_dict[a] > c.MIN_ATOM_FREQUENCY]

    atom_token_dict = {a: Chem.Atom(a).GetAtomicNum() for a in token_atoms}

    assert atom_token_dict == c.ATOM_TOKEN_DICT
