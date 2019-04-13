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
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from sklearn.metrics import r2_score
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
TARGET_LIST = ['GATS3e']

use_cuda = torch.cuda.is_available() and USE_CUDA
seed_random_state(RAND_STATE)
device = torch.device('cuda: 1' if use_cuda else 'cpu')
print(f'Training on device {device}')

# Get the trn/val/tst datasets and dataloaders ################################
cid_smiles_csv_path = c.PCBA_CID_SMILES_CSV_PATH \
    if PCBA_ONLY else c.PC_CID_SMILES_CSV_PATH
cid_smiles_df = pd.read_csv(cid_smiles_csv_path,
                            sep='\t',
                            header=0,
                            index_col=0,
                            dtype=str)
cid_smiles_df.index = cid_smiles_df.index.map(str)
cid_smiles_dict = cid_smiles_df.to_dict()['SMILES']

cid_dscrptr_csv_path = c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH
cid_dscrptr_df = pd.read_csv(cid_dscrptr_csv_path,
                             sep='\t',
                             header=0,
                             index_col=0,
                             usecols=['CID'] + TARGET_LIST,
                             dtype={t: np.float32 for t in TARGET_LIST})
cid_dscrptr_df.index = cid_dscrptr_df.index.map(str)

# Perform STD normalization for multi-target regression
cid_dscrptr_df_mean = cid_dscrptr_df.mean().values
cid_dscrptr_df_std = cid_dscrptr_df.std().values
cid_dscrptr_df = \
    (cid_dscrptr_df - cid_dscrptr_df.mean()) / cid_dscrptr_df.std()

cid_dscrptr_dict = cid_dscrptr_df.to_dict()

# List of CIDs for training, validation, and testing
# Make sure that all entries in the CID list is valid
smiles_cid_set = set(list(cid_smiles_dict.keys()))
dscrptr_cid_set = set(list(cid_dscrptr_dict[TARGET_LIST[0]].keys()))
cid_list = sorted(list(smiles_cid_set & dscrptr_cid_set), key=int)

trn_cid_list, tst_cid_list = train_test_split(cid_list,
                                              test_size=TEST_SIZE,
                                              random_state=RAND_STATE)
trn_cid_list, val_cid_list = train_test_split(trn_cid_list,
                                              test_size=VALIDATION_SIZE,
                                              random_state=RAND_STATE)

# Datasets and dataloaders
dataset_kwargs = {
    'target_list': TARGET_LIST,
    'cid_smiles_dict': cid_smiles_dict,
    'cid_dscrptr_dict': cid_dscrptr_dict}
trn_dataset = GraphToDscrptrDataset(cid_list=trn_cid_list, **dataset_kwargs)
val_dataset = GraphToDscrptrDataset(cid_list=val_cid_list, **dataset_kwargs)
tst_dataset = GraphToDscrptrDataset(cid_list=tst_cid_list, **dataset_kwargs)

dataloader_kwargs = {
    'batch_size': 64,
    'timeout': 1,
    'pin_memory': True if use_cuda else False,
    'num_workers': 16 if use_cuda else 0}
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
    optimizer, mode='min', factor=0.8, patience=5, min_lr=0.00001)


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
    sum_mae = 0.

    trgt_array, pred_array = np.array([]), np.array([])

    for data in loader:

        data = data.to(device)
        pred = model(data)

        sum_mae += (pred - data.y).abs().sum().item()  # MAE

        trgt_array = np.concatenate(
            (trgt_array, data.y.cpu().numpy().flatten()))
        pred_array = np.concatenate(
            (pred_array, pred.cpu().detach().numpy().flatten()))

    mae = sum_mae / len(loader.dataset)
    r2 = r2_score(y_pred=pred_array, y_true=trgt_array)
    return r2, mae


best_val_r2 = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_r2, val_mae = test(val_loader)
    scheduler.step(val_r2)

    if best_val_r2 is None or val_r2 > best_val_r2:
        best_val_r2 = val_r2
        tst_r2, tst_mae = test(tst_loader)

    print('Epoch: {:03d}, LR: {:6f}, Loss: {:.4f}, '.format(epoch, lr, loss),
          'Validation R2: {:.3f} MAE: {:.4f}; '.format(val_r2, val_mae),
          'Testing R2: {:.3f} MAE: {:.4f};'.format(tst_r2, tst_mae))

