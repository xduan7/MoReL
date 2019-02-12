""" 
    File Name:          MoReL/data_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        Prepare training/validation data.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

import utils.data_prep.config as c
from utils.data_prep.download import download


def data_prep():

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
    cid_mol_list = []

    for chunk_idx, cid_inchi_df_chunk in enumerate(
            pd.read_csv(c.CID_INCHI_FILE_PATH,
                        sep='\t',
                        header=None,
                        index_col=[0],
                        usecols=[0, 1],
                        chunksize=2 ** 12)):

        for cid, row in cid_inchi_df_chunk.iterrows():

            if cid not in pcba_cid_set:
                continue

            inchi = row[1]
            try:
                mol = Chem.MolFromInchi(inchi)
                assert mol
            except AssertionError:
                # print('Failed converting compound (CID=%i) to Mol.' % cid)
                unused_cid.append(cid)
                continue

            cid_mol_list.append([cid, mol])

    with open(c.UNUSED_CID_FILE_PATH, 'w') as f:
        json.dump(unused_cid, f, indent=4)

    cid_mol_df = pd.DataFrame(cid_mol_list, columns=['CID', 'Mol'])
    cid_mol_df.set_index('CID', inplace=True)
    cid_mol_df.to_hdf(c.CID_MOL_FILE_PATH, key='RDkit Molecule')

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
