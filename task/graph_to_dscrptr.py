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
from network.gnn.mpnn.mpnn import MPNN
from utils.misc.random_seeding import seed_random_state
from utils.dataset.graph_to_dscrptr_dataset import GraphToDscrptrDataset


# Constants and initializations ###############################################
# TODO: Need to incorporate constants into argparse
PCBA_ONLY = True
USE_CUDA = True
RAND_STATE = 0
TEST_SIZE = 10000
VALIDATION_SIZE = 10000
TARGET_LIST = c.TARGET_D7_DSCRPTR_NAMES[:20]

use_cuda = torch.cuda.is_available() and USE_CUDA
seed_random_state(RAND_STATE)
device = torch.device('cuda: 1' if use_cuda else 'cpu')
print(f'Training on device {device}')

# Get the trn/val/tst dataset and dataloaders ################################
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
dscrptr_mean = cid_dscrptr_df.mean().values
dscrptr_std = cid_dscrptr_df.std().values
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

# Downsizing training set for the purpose of testing
# _, trn_cid_list = train_test_split(trn_cid_list,
#                                    test_size=VALIDATION_SIZE * 10,
#                                    random_state=RAND_STATE)

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
    'num_workers': 8 if use_cuda else 0}
trn_loader = pyg_data.DataLoader(trn_dataset,
                                 shuffle=True,
                                 **dataloader_kwargs)
val_loader = pyg_data.DataLoader(val_dataset,
                                 **dataloader_kwargs)
tst_loader = pyg_data.DataLoader(tst_dataset,
                                 **dataloader_kwargs)


# Model, optimizer, and scheduler #############################################
model = MPNN(node_attr_dim=trn_dataset.node_attr_dim,
             edge_attr_dim=trn_dataset.edge_attr_dim,
             state_dim=128,
             num_conv=6,
             out_dim=len(TARGET_LIST)).to(device)
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


def test(loader, validation=True):
    model.eval()
    mae_array = np.zeros(shape=(len(TARGET_LIST)))
    trgt_array = np.zeros(shape=(0, len(TARGET_LIST)))
    pred_array = np.zeros(shape=(0, len(TARGET_LIST)))

    for data in loader:

        data = data.to(device)
        pred = model(data)

        # mae_array += (pred - data.y).abs().sum(dim=-1).item() * dscrptr_std

        trgt = data.y.cpu().numpy().reshape(-1, len(TARGET_LIST))
        pred = pred.detach().cpu().numpy().reshape(-1, len(TARGET_LIST))

        trgt = trgt * dscrptr_std + dscrptr_mean
        pred = pred * dscrptr_std + dscrptr_mean

        trgt_array = np.vstack((trgt_array, trgt))
        pred_array = np.vstack((pred_array, pred))
        mae_array += np.sum(np.abs(trgt - pred), axis=0)

    mae_array = mae_array / len(loader.dataset)

    # Save the results
    if not validation:
        np.save(c.PROCESSED_DATA_DIR + '/pred_array.npy', pred_array)
        np.save(c.PROCESSED_DATA_DIR + '/trgt_array.npy', trgt_array)

    r2_array = np.array(
        [r2_score(y_pred=pred_array[:, i], y_true=trgt_array[:, i])
         for i, t in enumerate(TARGET_LIST)])

    for i, target in enumerate(TARGET_LIST):
        print(f'Target Descriptor Name: {target:15s}, '
              f'R2: {r2_array[i]:.4f}, MAE: {mae_array[i]:.4f}')

    return np.mean(r2_array), np.mean(mae_array)


best_val_r2 = None
for epoch in range(1, 301):

    scheduler.step()
    loss = train(epoch)
    print('Validation ' + '#' * 80)
    val_r2, val_mae = test(val_loader)
    print('#' * 80)

    if best_val_r2 is None or val_r2 > best_val_r2:
        best_val_r2 = val_r2
        print('Testing ' + '#' * 80)
        tst_r2, tst_mae = test(tst_loader, validation=False)
        print('#' * 80)

    print('Epoch: {:03d}, LR: {:6f}, Loss: {:.4f}, '.format(epoch, lr, loss),
          'Validation R2: {:.4f} MAE: {:.4f}; '.format(val_r2, val_mae),
          'Testing R2: {:.4f} MAE: {:.4f};'.format(tst_r2, tst_mae))

