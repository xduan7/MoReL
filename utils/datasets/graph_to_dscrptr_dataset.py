""" 
    File Name:          MoReL/graph_to_dscrptr_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/9/19
    Python Version:     3.5.4
    File Description:   

"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

import utils.data_prep.config as c
from utils.data_prep.featurizers import inchi_to_mol, mol_to_graph


class GraphToDscrptrDataset(Dataset):

    def __init__(self,
                 target_dscrptr: str,
                 cid_list: list = None,
                 pcba_only: bool = True,
                 master_atom: bool = True,
                 master_bond: bool = True,
                 max_num_atoms: int = 128,
                 atom_feat_list: list = None,
                 bond_feat_list: list = None,
                 cid_inchi_hdf5: h5py.File = None,
                 cid_dscrptr_hdf5: h5py.File = None):

        super().__init__()
        self.__master_atom = master_atom
        self.__master_bond = master_bond
        self.__max_num_atoms = max_num_atoms
        self.__atom_feat_list = atom_feat_list
        self.__bond_feat_list = bond_feat_list

        # First load the hdf5 files if not given ##############################
        cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH \
            if pcba_only else c.PC_CID_INCHI_HDF5_PATH
        self.__cid_inchi_hdf5 = cid_inchi_hdf5 if cid_inchi_hdf5 else \
            h5py.File(cid_inchi_hdf5_path, 'r',  libver='latest')

        cid_dscrptr_hdf5_path = c.PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH
        self.__cid_dscrptr_hdf5 = cid_dscrptr_hdf5 if cid_dscrptr_hdf5 else \
            h5py.File(cid_dscrptr_hdf5_path, 'r', libver='latest')

        # Get the target descriptor index #####################################
        # TODO: if the following list indexing failed, the dataset should
        #  probably load the full descriptor file
        dscrptr_list = [dn.decode('UTF-8') for dn in
                        self.__cid_dscrptr_hdf5.get(name='DSCRPTR_NAMES')]
        self.__target_index = dscrptr_list.index(target_dscrptr)

        # Check the cid_list and eliminate invalid entries ####################
        # Make sure that the argument cid list are all strings
        if cid_list is not None:
            # If the list of CIDs are given, we trust them to be valid
            self.__cid_list = cid_list
        else:
            inchi_cid_set = set(list(self.__cid_inchi_hdf5.keys()))
            dscrptr_cid_set = set(list(self.__cid_dscrptr_hdf5.keys()))
            # Sort the CID list to make sure that we didn't introduce extra
            # randomness, and the results are easily reproducible
            self.__cid_list = \
                sorted(list(inchi_cid_set & dscrptr_cid_set), key=int)

        self.__len = len(self.__cid_list)

    def __len__(self):
        return self.__len

    def __getitem__(self, index: int):

        cid = self.get_cid(index)

        # Graph features, including nodes and edges features and adj matrix
        inchi = str(np.array(self.__cid_inchi_hdf5.get(name=cid)))
        n, adj, e = mol_to_graph(mol=inchi_to_mol(inchi),
                                 master_atom=self.__master_atom,
                                 master_bond=self.__master_bond,
                                 max_num_atoms=self.__max_num_atoms,
                                 atom_feat_list=self.__atom_feat_list,
                                 bond_feat_list=self.__bond_feat_list)

        # Target descriptor
        target = np.array([self.__cid_dscrptr_hdf5.get(
            name=cid)[self.__target_index]], dtype=np.float32)

        return Data(x=torch.from_numpy(n),
                    edge_index=torch.from_numpy(adj),
                    edge_attr=torch.from_numpy(e),
                    y=torch.from_numpy(target))

    def get_cid(self, index: int) -> str:
        return self.__cid_list[index]


if __name__ == '__main__':
    ds = GraphToDscrptrDataset(target_dscrptr='CIC5')
    d = ds[0]
