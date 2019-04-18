""" 
    File Name:          MoReL/graph_to_dscrptr_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/9/19
    Python Version:     3.5.4
    File Description:   

"""
import multiprocessing

import h5py
import torch
import logging
import numpy as np
import pandas as pd
from ilock import ILock
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Optional

import utils.data_prep.config as c
from utils.data_prep.featurizers import mol_to_graph

logger = logging.getLogger(__name__)


class GraphToDscrptrDataset(Dataset):

    def __init__(self,
                 target_list: iter,
                 cid_list: list = None,
                 pcba_only: bool = True,
                 master_atom: bool = True,
                 master_bond: bool = True,
                 max_num_atoms: int = 128,
                 atom_feat_list: list = None,
                 bond_feat_list: list = None,
                 cid_smiles_dict: dict = None,
                 cid_dscrptr_dict: dict = None):

        super().__init__()
        self.__target_list = target_list
        self.__master_atom = master_atom
        self.__master_bond = master_bond
        self.__max_num_atoms = max_num_atoms
        self.__atom_feat_list = atom_feat_list
        self.__bond_feat_list = bond_feat_list

        # First load the csv files into dict if not given #####################
        if cid_smiles_dict is None:
            cid_smiles_csv_path = c.PCBA_CID_SMILES_CSV_PATH \
                if pcba_only else c.PC_CID_SMILES_CSV_PATH
            cid_smiles_df = pd.read_csv(cid_smiles_csv_path,
                                        sep='\t',
                                        header=0,
                                        index_col=0,
                                        dtype=str)
            cid_smiles_df.index = cid_smiles_df.index.map(str)
            cid_smiles_dict = cid_smiles_df.to_dict()['SMILES']
        self.__cid_smiles_dict = cid_smiles_dict

        if cid_dscrptr_dict is None:
            # cid_dscrptr_csv_path = c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH
            # cid_dscrptr_df = pd.read_csv(
            #     cid_dscrptr_csv_path,
            #     sep='\t',
            #     header=0,
            #     index_col=0,
            #     usecols=['CID'] + self.__target_list,
            #     dtype={t: np.float32 for t in self.__target_list})
            # cid_dscrptr_df.index = cid_dscrptr_df.index.map(str)
            # self.__cid_dscrptr_dict = cid_dscrptr_df.to_dict()
            cid_list = []
            dscrptr_array = np.array([]).reshape(0, len(self.__target_list))
            for chunk_cid_dscrptr_df in pd.read_csv(
                    c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH,
                    sep='\t',
                    header=0,
                    index_col=0,
                    usecols=['CID'] + self.__target_list,
                    dtype={**{'CID': str},
                           **{t: np.float32 for t in self.__target_list}},
                    chunksize=2 ** 16):
                chunk_cid_dscrptr_df.index = chunk_cid_dscrptr_df.index.map(
                    str)
                cid_list.extend(list(chunk_cid_dscrptr_df.index))
                dscrptr_array = np.vstack(
                    (dscrptr_array, chunk_cid_dscrptr_df.values))

            # Perform STD normalization for multi-target regression
            dscrptr_mean = np.mean(dscrptr_array, axis=0)
            dscrptr_std = np.std(dscrptr_array, axis=0)
            dscrptr_array = (dscrptr_array - dscrptr_mean) / dscrptr_std

            assert len(cid_list) == len(dscrptr_array)
            self.__cid_dscrptr_dict = \
                {cid: dscrptr for cid, dscrptr in zip(cid_list, dscrptr_array)}

        else:
            # self.__cid_dscrptr_dict = {k: v for k, v
            #                            in cid_dscrptr_dict.items()
            #                            if k in self.__target_list}
            self.__cid_dscrptr_dict = cid_dscrptr_dict

        # Old code for HDF5 file processing
        # cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH \
        #     if pcba_only else c.PC_CID_INCHI_HDF5_PATH
        # self.__cid_inchi_hdf5 = cid_inchi_hdf5 if cid_inchi_hdf5 else \
        #     h5py.File(cid_inchi_hdf5_path, 'r',  libver='latest')
        #
        # cid_dscrptr_hdf5_path = c.PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH
        # self.__cid_dscrptr_hdf5 = cid_dscrptr_hdf5 if cid_dscrptr_hdf5 else \
        #     h5py.File(cid_dscrptr_hdf5_path, 'r', libver='latest')

        # Check the cid_list and eliminate invalid entries ####################
        # Make sure that the argument cid list are all strings
        if cid_list is not None:
            # If the list of CIDs are given, we trust them to be valid
            self.__cid_list = cid_list
        else:
            smiles_cid_set = set(list(self.__cid_smiles_dict.keys()))
            dscrptr_cid_set = set(
                list(self.__cid_dscrptr_dict[self.__target_list[0]].keys()))
            # Sort the CID list to make sure that we didn't introduce extra
            # randomness, and the results are easily reproducible
            self.__cid_list = \
                sorted(list(smiles_cid_set & dscrptr_cid_set), key=int)

        # Properties for dataset ##############################################
        self.__len = len(self.__cid_list)

        # This part is implemented awkwardly, to change this we need some
        # changes on the featurizer, make it a class or something
        single_data = self[0]
        self.node_attr_dim = single_data.x.shape[1]
        self.edge_attr_dim = single_data.edge_attr.shape[1]

    def __len__(self):
        return self.__len

    def __getitem__(self, index: int):

        # TODO: This read is not thread-safe
        # We can either protect this region with lock/mutex
        # Or, using literally anything but HDF5
        cid = self.get_cid(index)

        # Graph features, including nodes and edges features and adj matrix
        smiles = self.__cid_smiles_dict[cid]
        mol = Chem.MolFromSmiles(smiles)
        n, adj, e = mol_to_graph(mol=mol,
                                 master_atom=self.__master_atom,
                                 master_bond=self.__master_bond,
                                 max_num_atoms=self.__max_num_atoms,
                                 atom_feat_list=self.__atom_feat_list,
                                 bond_feat_list=self.__bond_feat_list)

        # Target descriptors
        # target = np.array([self.__cid_dscrptr_dict[t][cid] for t in
        #                    self.__target_list], dtype=np.float32)
        target = self.__cid_dscrptr_dict[cid]

        return Data(x=torch.from_numpy(n),
                    edge_index=torch.from_numpy(adj),
                    edge_attr=torch.from_numpy(e),
                    y=torch.from_numpy(target))

    def get_cid(self, index: int) -> str:
        return self.__cid_list[index]

    def get_index(self, cid: str) -> Optional[int]:
        try:
            return self.__cid_list.index(cid)
        except ValueError:
            return None


if __name__ == '__main__':
    ds = GraphToDscrptrDataset(target_list=['MW', 'AMW'])
    d = ds[0]
