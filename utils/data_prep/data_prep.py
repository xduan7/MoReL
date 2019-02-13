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
import codecs
import json
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

import utils.data_prep.config as c
from utils.data_prep.download import download


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
    # Save all the molecules as intermediate results

    unused_cid = []

    cid_mol_hdf5 = pd.HDFStore(c.CID_MOL_FILE_PATH)
    min_itemsize = 0

    for chunk_idx, chunk_cid_inchi_df in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 12)):
        chunk_cid_mol_list = []

        for cid, row in chunk_cid_inchi_df.iterrows():

            # Skip this compound if it is not in PCBA
            if cid not in pcba_cid_set:
                continue

            inchi = row[1]
            try:
                mol: Chem.rdchem.Mol = Chem.MolFromInchi(inchi)
                assert mol

                num_atoms: int = mol.GetNumAtoms()
                len_smiles: int = len(Chem.MolToSmiles(mol))

                # Skip the molecules with too many atoms or lengthy SMILES
                if num_atoms > c.MAX_NUM_ATOM or len_smiles > c.MAX_LEN_SMILES:
                    continue

            except AssertionError:
                # print('Failed converting compound (CID=%i) to Mol.' % cid)
                unused_cid.append(cid)
                continue

            # Serialize and encode the molecule object
            mol_str = mol_to_str(mol)
            min_itemsize = max(min_itemsize, len(mol_str))

            if len(mol_str) > c.MAX_LEN_MOL_STR:
                # print('Compund %s has %i number of characters after encoding.'
                #       % (Chem.MolToSmiles(mol), len(enc_mol)))
                unused_cid.append(cid)
                continue

            chunk_cid_mol_list.append([cid, mol_str])

        chunk_cid_mol_df = pd.DataFrame(chunk_cid_mol_list,
                                        columns=['CID', 'Mol'])
        chunk_cid_mol_df.set_index('CID', inplace=True)
        cid_mol_hdf5.append(key='RDkit Molecule',
                            value=chunk_cid_mol_df,
                            min_itemsize=c.MAX_LEN_MOL_STR)

    with open(c.UNUSED_CID_FILE_PATH, 'w') as f:
        json.dump(unused_cid, f, indent=4)

    print('The maximum length of encoded molecule is %i.' % min_itemsize)

    return


if __name__ == '__main__':



    data_prep()

    # df = pd.DataFrame(pd.read_hdf(c.CID_MOL_FILE_PATH))
    # print(type(df))
    # print(df.head())
    # a = df.values
    #
    # print(a[-1, -1])
    #
    # s = str(a[-1, -1])
    #
    # Chem.Mol(s)

    # for cid, binary_str in df.iterrows():
    #
    #     print(cid)
    #     print(binary_str)
