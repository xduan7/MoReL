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
import torch
import h5py
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from sklearn.model_selection import train_test_split

import utils.data_prep.config as c
from networks.gnn.mpnn.mpnn import MPNN
from utils.misc.random_seeding import seed_random_state
from utils.datasets.graph_to_dscrptr_dataset import GraphToDscrptrDataset


# Constants and initializations ###############################################
# TODO: Need to incorporate constants into argparse
PCBA_ONLY = True
USE_CUDA = True
RAND_STATE = 0
TEST_SIZE = 10000
VALIDATION_SIZE = 10000
TARGET_DSCRPTR = 'CIC5'

use_cuda = torch.cuda.is_available() and USE_CUDA
seed_random_state(RAND_STATE)
device = torch.device('cuda: 1' if use_cuda else 'cpu')
print(f'Training on device {device}')

# Get the trn/val/tst datasets and dataloaders ################################
# HDF5 files
cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH \
    if PCBA_ONLY else c.PC_CID_INCHI_HDF5_PATH
cid_inchi_hdf5 = h5py.File(cid_inchi_hdf5_path, 'r',  libver='latest')

cid_dscrptr_hdf5_path = c.PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH
cid_dscrptr_hdf5 = h5py.File(cid_dscrptr_hdf5_path, 'r', libver='latest')

# List of CIDs for training, validation, and testing
# Make sure that all entries in the CID list is valid
inchi_cid_set = set(list(cid_inchi_hdf5.keys()))
dscrptr_cid_set = set(list(cid_dscrptr_hdf5.keys()))
cid_list = sorted(list(inchi_cid_set & dscrptr_cid_set), key=int)

trn_cid_list, tst_cid_list = train_test_split(cid_list,
                                              test_size=TEST_SIZE,
                                              random_state=RAND_STATE)
trn_cid_list, val_cid_list = train_test_split(trn_cid_list,
                                              test_size=VALIDATION_SIZE,
                                              random_state=RAND_STATE)

# Sample the training CID list
_, trn_cid_list = train_test_split(trn_cid_list,
                                   test_size=VALIDATION_SIZE * 5,
                                   random_state=RAND_STATE)


# Datasets and dataloaders
dataset_kwargs = {
    'target_dscrptr': TARGET_DSCRPTR,
    'cid_inchi_hdf5': cid_inchi_hdf5,
    'cid_dscrptr_hdf5': cid_dscrptr_hdf5}
trn_dataset = GraphToDscrptrDataset(cid_list=trn_cid_list, **dataset_kwargs)
val_dataset = GraphToDscrptrDataset(cid_list=val_cid_list, **dataset_kwargs)
tst_dataset = GraphToDscrptrDataset(cid_list=tst_cid_list, **dataset_kwargs)

dataloader_kwargs = {
    'batch_size': 64,
    'timeout': 1,
    'pin_memory': True if use_cuda else False,
    'num_workers': 1 if use_cuda else 0}
trn_loader = pyg_data.DataLoader(trn_dataset,
                                 shuffle=True,
                                 **dataloader_kwargs)
val_loader = pyg_data.DataLoader(val_dataset,
                                 **dataloader_kwargs)
tst_loader = pyg_data.DataLoader(tst_dataset,
                                 **dataloader_kwargs)


# Model, optimizer, and scheduler #############################################
model = MPNN(node_attr_dim=trn_dataset.node_attr_dim,
             edge_attr_dim=trn_dataset.edge_attr_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in trn_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(trn_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) - data.y).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(tst_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f},'.format(epoch, lr, loss, val_error, test_error))

