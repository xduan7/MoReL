""" 
    File Name:          MoReL/combo_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/28/19
    Python Version:     3.5.4
    File Description:   

"""
import pickle
import time
import h5py
import numpy as np
import pandas as pd
import torch.utils.data as data
from rdkit import Chem
from argparse import Namespace
from mmap import mmap
from multiprocessing.managers import DictProxy

from sklearn.model_selection import train_test_split

import utils.data_prep.config as c
from utils.data_prep.data_prep import get_from_hdf5
from utils.data_prep.ecfp_prep import mol_to_ecfp
from utils.data_prep.graph_prep import mol_to_graph
from utils.data_prep.mol_prep import str_to_mol
from utils.data_prep.smiles_prep import mol_to_token


class ComboDataset(data.Dataset):
    """
    TODO: performance issue with dataloader
    One big issue is that, during the dataloader constructing, training and
    testing dataloader will go through the exact same process until
    train/test split. We can probably save more than a couple of seconds
    just by sharing the file handlers and shared memory structures between
    training and testing dataloader.

    """
    def __init__(self,
                 args: Namespace,
                 training: bool = True,
                 shared_dict: DictProxy or mmap = None):

        super().__init__()

        # Load the features ###################################################
        # Type of feature that it returns
        # Must be one of 'token', 'ecfp', 'graph'
        self.__feature_type = args.feature_type.lower()

        # Featurization strategy
        # Depending on the strategy, we can do one of the followings:
        # (1)'loading'; (2)'computing'; (3)'dict_proxy'; (4) 'mmap'
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
        self.__cid_mol_str_hdf5 = \
            h5py.File(c.CID_MOL_STR_HDF5_PATH, 'r', libver='latest')

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

        # Shared data structure for features
        if self.__featurization == 'dict_proxy' \
                or self.__featurization == 'mmap':
            assert shared_dict
            self.__shared_dict = shared_dict
            self.__dict_timeout: int = args.dict_timeout

        # Load the target (dragon7 descriptor) ################################
        cid_target_array = pd.read_csv(
            c.PCBA_CID_TARGET_DSCPTR_FILE_PATH,
            header=0,
            na_values='na',
            dtype=str,
            usecols=['CID', args.target_dscrptr_name]).as_matrix()

        self.__cid_target_dict = {}
        for cid, target in cid_target_array:
            try:
                target = np.float32(target)
                assert (not np.isnan(target))
            except (ValueError, AssertionError):
                continue

            self.__cid_target_dict[cid] = target

        # Train/test split ####################################################
        # Get the list of valid CIDs, and construct a mapping (dict)
        # From index: int -> CID: str
        # Sort the list for deterministic ordering
        cid_list = sorted(list(
            set(self.__cid_mol_str_hdf5.keys())
            & set(self.__cid_target_dict.keys())), key=int)

        trn_cid_list, test_cid_list = train_test_split(
            cid_list, test_size=c.TEST_SIZE, random_state=args.rand_state)
        self.__cid_list = trn_cid_list if training else test_cid_list

        self.__index_cid_dict = \
            {i: cid for i, cid in enumerate(self.__cid_list)}
        self.__len = len(self.__cid_list)

        # Public variable that keeps track of the time spent on __getitem__
        self.getitem_time_ms = 0

    def __len__(self):
        return self.__len

    def __cid_to_feature(self, cid: str) -> np.array:
        mol: Chem.Mol = str_to_mol(
            str(get_from_hdf5(cid=cid, cid_grp=self.__cid_mol_str_hdf5)))
        if self.__feature_type == 'token':
            return mol_to_token(mol)
        elif self.__feature_type == 'ecfp':
            return mol_to_ecfp(mol)
        else:
            return mol_to_graph(mol)

    def __getitem__(self, index: int) -> np.array:

        start_ms = int(round(time.time() * 1000))
        cid: str = self.__index_cid_dict[index]

        if self.__featurization == 'loading':
            if self.__feature_type == 'token':
                feature = get_from_hdf5(cid=cid,
                                        cid_grp=self.__cid_token_hdf5_grp)
            elif self.__feature_type == 'ecfp':
                feature = get_from_hdf5(cid=cid,
                                        cid_grp=self.__cid_ecfp_hdf5_grp)
            else:
                feature = (get_from_hdf5(cid=cid,
                                         cid_grp=self.__cid_node_hdf5_grp),
                           get_from_hdf5(cid=cid,
                                         cid_grp=self.__cid_edge_hdf5_grp))

        elif self.__featurization == 'computing':
            feature = self.__cid_to_feature(cid)

        elif self.__featurization == 'dict_proxy':

            self.__shared_dict: DictProxy

            # Here we first check if the feature of cid is already computed
            # by other training instances. If so we simply use that feature.
            # Otherwise, we have to compute the feature, same as 'computing'
            feature_key = '[%s][%s]' % (cid, self.__feature_type)

            if feature_key in self.__shared_dict:
                _, feature = self.__shared_dict[feature_key]

            else:
                # Compute feature and add to the dict, along with timestamp
                feature = self.__cid_to_feature(cid)
                curr_time_sec = int(round(time.time()))
                self.__shared_dict[feature_key] = (curr_time_sec, feature)

                # Note that this is the first process that reaches here
                # It will take the responsibility to clean up the dict,
                # which means that older features will be deleted.
                for k in self.__shared_dict.keys():
                    t, _ = self.__shared_dict[k]
                    if (curr_time_sec - t) > self.__dict_timeout:
                        del self.__shared_dict[k]

        elif self.__featurization == 'mmap':

            self.__shared_dict: mmap

            # Similar to dict_proxy.
            # However, here we need to take care of the read and write of
            # mmap file, which contains bytes of up to size (c.MMAP_BYTE_SIZE)
            feature_key = '[%s][%s]' % (cid, self.__feature_type)

            self.__shared_dict.seek(0)
            curr_bytes: bytes = self.__shared_dict.read()
            shared_dict: dict = pickle.loads(curr_bytes)

            if feature_key in shared_dict:
                _, feature = shared_dict[feature_key]

            else:
                # Compute feature and add to the dict, along with timestamp
                feature = self.__cid_to_feature(cid)
                curr_time_sec = int(round(time.time()))
                shared_dict[feature_key] = (curr_time_sec, feature)

                # Note that this is the first process that reaches here
                # It will take the responsibility to clean up the dict,
                # which means that older features will be deleted.
                for k in shared_dict.keys():
                    t, _ = shared_dict[k]
                    if (curr_time_sec - t) > self.__dict_timeout:
                        del shared_dict[k]

                # Write new dict back to mmap
                new_bytes = pickle.dumps(shared_dict)
                self.__shared_dict.seek(0)
                self.__shared_dict.write(
                    new_bytes + b'\0' * (len(new_bytes) - len(curr_bytes)))

        # When __featurization is set to anything else
        # Compute the feature here
        # Note that all the computation is already tested and passed during
        # the gathering of Mol object, which means that none of the
        # following computation operations should raise errors.
        else:
            feature = self.__cid_to_feature(cid)

        # Keeps track of the time consumed in getitem for different strategies
        self.getitem_time_ms += (int(round(time.time() * 1000)) - start_ms)

        return feature


if __name__ == '__main__':

    # A simple test of dataloading
    ns = Namespace(feature_type='graph',
                   featurization='loading',
                   target_dscrptr_name=c.TARGET_DSCRPTR_NAMES[0],
                   rand_state=0)

    dset = ComboDataset(ns)
    n, e = dset[0]
    print('Dataset spent %i ms fetching feature.' % dset.getitem_time_ms)
