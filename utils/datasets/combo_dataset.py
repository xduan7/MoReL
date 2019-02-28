""" 
    File Name:          MoReL/combo_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/28/19
    Python Version:     3.5.4
    File Description:   

"""
import time

import h5py
import argparse

import torch.utils.data as data
from rdkit import Chem

import utils.data_prep.config as c
from utils.data_prep.ecfp_prep import mol_to_ecfp
from utils.data_prep.graph_prep import mol_to_graph
from utils.data_prep.mol_prep import str_to_mol
from utils.data_prep.smiles_prep import mol_to_token


class ComboDataset(data.Dataset):

    def __init__(self, args: argparse.Namespace):

        super().__init__()

        # Type of feature that it returns
        # Must be one of 'token', 'ecfp', ''graph
        self.__feature_type = args.feature_type.lower()

        # Featurization strategy
        # Depending on the strategy, we can do one of the followings:
        # (1)'loading'; (2)'computing'; (3)'mixed'
        # (1)   Load features from HDF5 directly;
        # (2)   Load Mol from HDF5, and extract features here;
        # (3)   Same as (2), but prior to that, we check if other training
        #       instances has already done so and stored in the memory
        #       somewhere. If so, we simply load the features from memory.
        #       Otherwise, we need to compute and save the features for
        #       other training instances.
        self.__featurization = args.featurization.lower()

        # Get the HDF5 file (either CID-Mol or CID-features)
        self.__cid_features_hdf5 = \
            h5py.File(c.CID_FEATURES_HDF5_PATH, 'r', libver='latest')
        self.__cid_mol_str_hdf5_grp = \
            self.__cid_features_hdf5.get(name='CID-Mol_str')

        # Load the feature groups
        self.__cid_token_hdf5_grp = \
            self.__cid_features_hdf5.get(name='CID-token')
        self.__cid_ecfp_hdf5_grp = \
            self.__cid_features_hdf5.get(name='CID-ECFP')
        self.__cid_graph_hdf5_grp = \
            self.__cid_features_hdf5.get(name='CID-graph')
        self.__cid_node_hdf5_grp = \
            self.__cid_graph_hdf5_grp.get(name='CID-node')
        self.__cid_edge_hdf5_grp = \
            self.__cid_graph_hdf5_grp.get(name='CID-edge')

        # Get the list of valid CIDs, and construct a mapping (dict)
        # From index: int -> CID: str
        cid_list = list(self.__cid_mol_str_hdf5_grp.keys())
        self.__index_cid_dict = {i: cid for i, cid in enumerate(cid_list)}

        self.__len = len(cid_list)

        # Public variable that keeps track of the time spent on __getitem__
        self.getitem_time_ms = 0

    def __len__(self):
        return self.__len

    def __getitem__(self, index):

        start_ms = int(round(time.time() * 1000))
        cid: str = self.__index_cid_dict[index]
        feature = None

        if self.__featurization == 'loading':
            if self.__feature_type == 'token':
                feature = self.__cid_token_hdf5_grp.get(cid)
            elif self.__feature_type == 'ecfp':
                feature = self.__cid_ecfp_hdf5_grp.get(cid)
            else:
                feature = (self.__cid_node_hdf5_grp.get(cid),
                           self.__cid_edge_hdf5_grp.get(cid))

        if self.__featurization == 'mixed':

            # Here we first check if the feature of cid is already computed
            # by other training instances. If so we simply use that feature.
            # Otherwise, we have to compute the feature, same as 'computing'

            feature = None
            # Check shared data, return if found.

        # Compute the feature here
        # Note that all the computation is already tested and passed during
        # the gathering of Mol object, which means that none of the
        # following computation operations should raise errors.
        if feature is None:

            mol: Chem.Mol = \
                str_to_mol(self.__cid_mol_str_hdf5_grp.get(cid))

            if self.__feature_type == 'token':
                feature = mol_to_token(mol)
            elif self.__feature_type == 'ecfp':
                feature = mol_to_ecfp(mol)
            else:
                feature = mol_to_graph(mol)

        # Keeps track of the time consumed in getitem for different strategies
        self.getitem_time_ms += (int(round(time.time() * 1000)) - start_ms)

        return feature


if __name__ == '__main__':

    # A simple test to compare
