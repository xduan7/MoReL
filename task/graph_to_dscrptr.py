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
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import sys
sys.path.extend(['/home/xduan7/Projects/MoReL'])
import utils.data_prep.config as c
from network.gnn.gat.gat import EdgeGATEncoder
from network.gnn.gcn.gcn import EdgeGCNEncoder
from network.gnn.mpnn.mpnn import MPNN
from utils.misc.random_seeding import seed_random_state
from utils.misc.parameter_counting import count_parameters
from utils.dataset.graph_to_dscrptr_dataset import GraphToDscrptrDataset


def main():

    parser = argparse.ArgumentParser(
        description='Graph Model for Dragon7 Descriptor Prediction')

    parser.add_argument('--model_type', type=str, default='mpnn',
                        help='type of convolutional graph model',
                        choices=['mpnn', 'gcn', 'gat'])
    parser.add_argument('--pooling', type=str, default='set2set',
                        help='global pooling layer for graph model',
                        choices=['set2set', 'attention'])
    parser.add_argument('--state_dim', type=int, default=256,
                        help='hidden state dimension for conv layers')
    parser.add_argument('--num_conv', type=int, default=3,
                        help='number of convolution operations')
    parser.add_argument('--num_dscrptr', type=int, default=100,
                        help='number of dragon7 descriptors for prediction')

    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization for nn weights')
    parser.add_argument('--lr_decay_patience', type=int, default=8,
                        help='decay patience for learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                        help='decay factor for learning rate')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='maximum number of epochs')

    parser.add_argument('--val_size', type=int or float, default=10000)
    parser.add_argument('--tst_size', type=int or float, default=10000)

    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()

    # Constants and initializations ###########################################
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device(f'cuda: {args.cuda_device}' if use_cuda else 'cpu')
    print(f'Training on device {device}')

    seed_random_state(args.rand_state)

    target_list = c.TARGET_D7_DSCRPTR_NAMES[: args.num_dscrptr]

    # Get the trn/val/tst dataset and dataloaders #############################
    print('Preparing CID-SMILES dictionary ... ')
    cid_smiles_csv_path = c.PCBA_CID_SMILES_CSV_PATH
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
    dscrptr_array = np.array([], dtype=np.float32).reshape(0, len(target_list))
    for chunk_cid_dscrptr_df in pd.read_csv(
            c.PCBA_CID_TARGET_D7DSCPTR_CSV_PATH,
            sep='\t',
            header=0,
            index_col=0,
            usecols=['CID'] + target_list,
            dtype={**{'CID': str}, **{t: np.float32 for t in target_list}},
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

    trn_cid_list, tst_cid_list = \
        train_test_split(cid_list,
                         test_size=args.tst_size,
                         random_state=args.rand_state)
    trn_cid_list, val_cid_list = \
        train_test_split(trn_cid_list,
                         test_size=args.val_size,
                         random_state=args.rand_state)

    # Downsizing training set for the purpose of testing
    _, trn_cid_list = train_test_split(trn_cid_list,
                                       test_size=args.val_size * 10,
                                       random_state=args.rand_state)

    # Datasets and dataloaders
    dataset_kwargs = {
        'target_list': target_list,
        'cid_smiles_dict': cid_smiles_dict,
        'cid_dscrptr_dict': cid_dscrptr_dict,
        # 'multi_edge_indices': (MODEL_TYPE.upper() == 'GCN') or
        #                       (MODEL_TYPE.upper() == 'GAT')
    }
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

    # Model, optimizer, and scheduler #########################################
    attention_pooling = (args.pooling == 'attention')

    if args.model_type.upper() == 'GCN':
        model = EdgeGCNEncoder(node_attr_dim=trn_dataset.node_attr_dim,
                               edge_attr_dim=trn_dataset.edge_attr_dim,
                               state_dim=args.state_dim,
                               num_conv=args.num_conv,
                               out_dim=len(target_list),
                               attention_pooling=attention_pooling).to(device)
    elif args.model_type.upper() == 'GAT':
        model = EdgeGATEncoder(node_attr_dim=trn_dataset.node_attr_dim,
                               edge_attr_dim=trn_dataset.edge_attr_dim,
                               state_dim=args.state_dim,
                               num_conv=args.num_conv,
                               out_dim=len(target_list),
                               attention_pooling=attention_pooling).to(device)
    else:
        model = MPNN(node_attr_dim=trn_dataset.node_attr_dim,
                     edge_attr_dim=trn_dataset.edge_attr_dim,
                     state_dim=args.state_dim,
                     num_conv=args.num_conv,
                     out_dim=len(target_list),
                     attention_pooling=attention_pooling).to(device)

    num_params = count_parameters(model)
    print(f'Model Summary (Number of Parameters: {num_params})\n{model}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.init_lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_factor,
        patience=args.lr_decay_patience, min_lr=1e-6)

    def train(loader):
        model.train()
        loss_all = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y.view(-1, len(target_list)))
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(trn_loader.dataset)

    def test(loader):
        model.eval()
        mae_array = np.zeros(shape=(len(target_list)))
        trgt_array = np.zeros(shape=(0, len(target_list)))
        pred_array = np.zeros(shape=(0, len(target_list)))

        for data in loader:

            data = data.to(device)
            pred = model(data)

            trgt = data.y.cpu().numpy().reshape(-1, len(target_list))
            pred = pred.detach().cpu().numpy().reshape(-1, len(target_list))

            trgt = trgt * dscrptr_std + dscrptr_mean
            pred = pred * dscrptr_std + dscrptr_mean

            trgt_array = np.vstack((trgt_array, trgt))
            pred_array = np.vstack((pred_array, pred))
            mae_array += np.sum(np.abs(trgt - pred), axis=0)

        mae_array = mae_array / len(loader.dataset)

        # # Save the results
        #     np.save(c.PROCESSED_DATA_DIR + '/pred_array.npy', pred_array)
        #     np.save(c.PROCESSED_DATA_DIR + '/trgt_array.npy', trgt_array)

        r2_array = np.array(
            [r2_score(y_pred=pred_array[:, i], y_true=trgt_array[:, i])
             for i, t in enumerate(target_list)])

        for i, target in enumerate(target_list):
            print(f'Target Descriptor Name: {target:15s}, '
                  f'R2: {r2_array[i]:.4f}, MAE: {mae_array[i]:.4f}')

        return np.mean(r2_array), np.mean(mae_array)

    print('Training started.')
    best_val_r2 = None
    for epoch in range(1, args.max_num_epochs + 1):

        # scheduler.step()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(trn_loader)
        print('Validation ' + '#' * 80)
        val_r2, val_mae = test(val_loader)
        print('#' * 80)
        scheduler.step(val_r2)

        if best_val_r2 is None or val_r2 > best_val_r2:
            best_val_r2 = val_r2
            print('Testing ' + '#' * 80)
            tst_r2, tst_mae = test(tst_loader)
            print('#' * 80)

        print(f'Epoch: {epoch:03d}, LR: {lr:6f}, Loss: {loss:.4f}, ',
              f'Validation R2: {val_r2:.4f} MAE: {val_mae:.4f}; ',
              f'Testing R2: {tst_r2:.4f} MAE: {tst_mae:.4f};')


if __name__ == '__main__':
    main()
