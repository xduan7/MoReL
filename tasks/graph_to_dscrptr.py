""" 
    File Name:          MoReL/graph_to_dscrptr.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/9/19
    Python Version:     3.5.4
    File Description:

        This task is highly based on the QM9 example in PyG:
        https://github.com/rusty1s/pytorch_geometric/blob/master/
        examples/qm9_nn_conv.py

        Note that there are several models that takes advantages of edge
        attributes:  gmm_conv, nn_conv, spline_conv

        In the meanwhile, we can still one-hot encode the edge attributes,
        and stack multiple graph models together (multi-graph approach).

        Related issues:
        https://github.com/rusty1s/pytorch_geometric/issues/147
        https://github.com/rusty1s/pytorch_geometric/issues/175
"""
import h5py
import utils.data_prep.config as c
from sklearn.model_selection import train_test_split

RAND_STATE = 0

# Get the trn/val/tst datasets and dataloaders ################################
cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH
cid_inchi_hdf5 = h5py.File(cid_inchi_hdf5_path, 'r',  libver='latest')

cid_dscrptr_hdf5_path = c.PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH
cid_dscrptr_hdf5 = h5py.File(cid_dscrptr_hdf5_path, 'r', libver='latest')

inchi_cid_set = set(list(cid_inchi_hdf5.keys()))
dscrptr_cid_set = set(list(cid_dscrptr_hdf5.keys()))
cid_list = sorted(list(inchi_cid_set & dscrptr_cid_set), key=int)

X_train, X_test, y_train, y_test\
    = train_test_split(cid_list, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val
= train_test_split(X_train, y_train, test_size=0.2, random_state=1)