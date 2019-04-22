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
from network.gnn.gcn.gcn import EdgeGCNEncoder
from network.gnn.mpnn.mpnn import MPNN
from utils.misc.random_seeding import seed_random_state
from utils.dataset.graph_to_dscrptr_dataset import GraphToDscrptrDataset


# Constants and initializations ###############################################
# TODO: Need to incorporate constants into argparse
from utils.misc.scheduler import CyclicCosAnnealingLR

PCBA_ONLY = True
USE_CUDA = True
RAND_STATE = 0
TEST_SIZE = 10000
VALIDATION_SIZE = 10000
TARGET_LIST = c.TARGET_D7_DSCRPTR_NAMES

use_cuda = torch.cuda.is_available() and USE_CUDA
seed_random_state(RAND_STATE)
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'Training on device {device}')

# Get the trn/val/tst dataset and dataloaders ################################
print('Preparing CID-SMILES dictionary ... ')
cid_smiles_csv_path = c.PCBA_CID_SMILES_CSV_PATH \
    if PCBA_ONLY else c.PC_CID_SMILES_CSV_PATH
cid_smiles_df = pd.read_csv(cid_smiles_csv_path,
                            sep='\t',
                            header=0,
                            index_col=0,
                            dtype=str)
cid_smiles_df.index = cid_smiles_df.index.map(str)
cid_smiles_dict = cid_smiles_df.to_dict()['SMILES']
del cid_smiles_df

print('Preparing CID-dscrptr dictionary ... ')
# cid_dscrptr_dict has a structure of dict[target_name][str(cid)]

# cid_dscrptr_df = pd.read_csv(c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH,
#                              sep='\t',
#                              header=0,
#                              index_col=0,
#                              usecols=['CID'] + TARGET_LIST,
#                              dtype={t: np.float32 for t in TARGET_LIST})
# cid_dscrptr_df.index = cid_dscrptr_df.index.map(str)
#
# # Perform STD normalization for multi-target regression
# dscrptr_mean = cid_dscrptr_df.mean().values
# dscrptr_std = cid_dscrptr_df.std().values
# cid_dscrptr_df = \
#     (cid_dscrptr_df - cid_dscrptr_df.mean()) / cid_dscrptr_df.std()
#
# cid_dscrptr_dict = cid_dscrptr_df.to_dict()
# del cid_dscrptr_df

cid_list = []
dscrptr_array = np.array([], dtype=np.float32).reshape(0, len(TARGET_LIST))
for chunk_cid_dscrptr_df in pd.read_csv(
        c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH,
        sep='\t',
        header=0,
        index_col=0,
        usecols=['CID'] + TARGET_LIST,
        dtype={**{'CID': str}, **{t: np.float32 for t in TARGET_LIST}},
        chunksize=2 ** 16):

    chunk_cid_dscrptr_df.index = chunk_cid_dscrptr_df.index.map(str)
    cid_list.extend(list(chunk_cid_dscrptr_df.index))
    dscrptr_array = np.vstack((dscrptr_array, chunk_cid_dscrptr_df.values))

# Perform STD normalization for multi-target regression
dscrptr_mean = np.mean(dscrptr_array, axis=0)
dscrptr_std = np.std(dscrptr_array, axis=0)
dscrptr_array = (dscrptr_array - dscrptr_mean) / dscrptr_std

assert len(cid_list) == len(dscrptr_array)
cid_dscrptr_dict = {cid: dscrptr
                    for cid, dscrptr in zip(cid_list, dscrptr_array)}


print('Preparing datasets and dataloaders ... ')
# List of CIDs for training, validation, and testing
# Make sure that all entries in the CID list is valid
smiles_cid_set = set(list(cid_smiles_dict.keys()))
dscrptr_cid_set = set(list(cid_dscrptr_dict.keys()))
cid_list = sorted(list(smiles_cid_set & dscrptr_cid_set), key=int)

trn_cid_list, tst_cid_list = train_test_split(cid_list,
                                              test_size=TEST_SIZE,
                                              random_state=RAND_STATE)
trn_cid_list, val_cid_list = train_test_split(trn_cid_list,
                                              test_size=VALIDATION_SIZE,
                                              random_state=RAND_STATE)

# # Downsizing training set for the purpose of testing
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
    'batch_size': 32,
    'timeout': 1,
    'pin_memory': True if use_cuda else False,
    'num_workers': 4 if use_cuda else 0}
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
             state_dim=256,
             num_conv=3,
             out_dim=len(TARGET_LIST)).to(device)
# model = EdgeGCNEncoder(node_attr_dim=trn_dataset.node_attr_dim,
#                        edge_attr_dim=trn_dataset.edge_attr_dim,
#                        state_dim=32,
#                        num_conv=3,
#                        out_dim=len(TARGET_LIST),
#                        attention_pooling=False).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)
# optimizer = torch.optim.SGD(model.parameters(),
#                             lr=1e-3,
#                             momentum=0.9,
#                             weight_decay=1e-4,
#                             nesterov=True)
# scheduler = CyclicCosAnnealingLR(optimizer,
#                                  milestones=[(2**i) * 4 for i in range(10)],
#                                  eta_min=1e-5)


def train(epoch):
    model.train()
    loss_all = 0

    for data in trn_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y.view(-1, len(TARGET_LIST)))
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


print('Training, validation, and testing ... ')
best_val_r2 = None
for epoch in range(1, 301):

    # scheduler.step()
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    print('Validation ' + '#' * 80)
    val_r2, val_mae = test(val_loader)
    print('#' * 80)
    scheduler.step(val_r2)

    if best_val_r2 is None or val_r2 > best_val_r2:
        best_val_r2 = val_r2
        print('Testing ' + '#' * 80)
        tst_r2, tst_mae = test(tst_loader, validation=False)
        print('#' * 80)

    print('Epoch: {:03d}, LR: {:6f}, Loss: {:.4f}, '.format(epoch, lr, loss),
          'Validation R2: {:.4f} MAE: {:.4f}; '.format(val_r2, val_mae),
          'Testing R2: {:.4f} MAE: {:.4f};'.format(tst_r2, tst_mae))

