import torch
import argparse
import numpy as np
from sklearn.metrics import r2_score
from torch import nn
import torch.nn.functional as F
from typing import List

import sys
sys.path.extend(['/raid/xduan7/Projects/MoReL'])
from utils.misc.random_seeding import seed_random_state
from utils.dataset.drug_resp_dataset import DrugRespDataset, \
    trim_resp_array, \
    get_resp_array, ScalingMethod, NanProcessing, DrugFeatureType, \
    CellProcessingMethod, CellSubsetType, CellDataType, get_datasets, \
    SubsampleType, DATA_SOURCES


# Simple Uno-like model
class SimpleUno(nn.Module):

    def __init__(self,
                 cell_dim: int,
                 drug_dim: int,
                 state_dim: int = 512,
                 dropout: float = 0.2):

        super(SimpleUno, self).__init__()

        self.__cell_tower = nn.Sequential(
            nn.Linear(cell_dim, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.__drug_tower = nn.Sequential(
            nn.Linear(drug_dim, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4096, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.__final_tower = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, state_dim, bias=True),
            nn.BatchNorm1d(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(state_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1, bias=True))

    def forward(self, cell_feature, drug_feature, concentration):

        return self.__final_tower(
            torch.cat((self.__cell_tower(cell_feature),
                       self.__drug_tower(drug_feature),
                       concentration), dim=-1))


def get_cross_study_datasets(
        trn_sources: List[str],
        tst_sources: List[str],
        subsample_on: str,
        subsample_percentage: float):

    trn_dset, _, trn_cell_dict, trn_drug_dict = get_datasets(
        resp_data_path='/raid/xduan7/Data/combined_single_drug_response.csv',
        resp_aggregated=False,
        resp_target='GROWTH',
        resp_data_sources=trn_sources,

        cell_data_dir='/raid/xduan7/Data/cell/',
        cell_data_type=CellDataType.RNASEQ,
        cell_subset_type=CellSubsetType.LINCS1000,
        cell_processing_method=CellProcessingMethod.SOURCE_SCALE,
        cell_scaling_method=ScalingMethod.NONE,
        cell_type_subset=None,

        drug_data_dir='/raid/xduan7/Data/drug/',
        drug_feature_type=DrugFeatureType.DRAGON7_DESCRIPTOR,
        drug_nan_processing=NanProcessing.DELETE_COL,
        drug_scaling_method=ScalingMethod.STANDARD,

        rand_state=0,
        test_ratio=0.,
        disjoint_drugs=False,
        disjoint_cells=False)

    tst_dsets = []
    for _tst_src in tst_sources:
        _tmp_resp_array = get_resp_array(
            data_path='/raid/xduan7/Data/combined_single_drug_response.csv',
            aggregated=False,
            target='GROWTH',
            data_sources=[_tst_src, ])
        _tmp_resp_array = trim_resp_array(
            resp_array=_tmp_resp_array,
            cells=trn_cell_dict.keys(),
            drugs=trn_drug_dict.keys(),
            inclusive=True)
        _tst_dset = DrugRespDataset(
            cell_dict=trn_cell_dict,
            drug_dict=trn_drug_dict,
            resp_array=_tmp_resp_array,
            aggregated=False)
        tst_dsets.append(_tst_dset)

    # Subsample the training set either on drug or cell
    subsample_type = SubsampleType(subsample_on)
    trn_dset.subsample(subsample_type, subsample_percentage)

    return trn_dset, tst_dsets


def run_instance(
        trn_sources: List[str],
        tst_sources: List[str],
        state_dim: int,
        subsample_on: str,
        subsample_percentage: float,
        device: torch.device):

    print('\n' + '#' * 80)
    print('#' * 80)

    print(f'Training Sources: {trn_sources} '
          f'(using only {subsample_percentage * 100: .0f}%% {subsample_on})')

    trn_dset, tst_dsets = \
        get_cross_study_datasets(trn_sources=trn_sources,
                                 tst_sources=tst_sources,
                                 subsample_on=subsample_on,
                                 subsample_percentage=subsample_percentage)

    print('Datasets Summary:')
    print('-' * 80)
    print(f'Training Dataset ({trn_sources}):')
    print(trn_dset)

    print('-' * 80)
    print('Testing Dataset(s):')
    for _i, _tst_dset in enumerate(tst_dsets):
        print(f'Data Source [{tst_sources[_i]}]')
        print(_tst_dset)

    print('-' * 80)
    print('#' * 80)

    # Get the dimensions of features in the most awkward way possible
    _src, _cell, _drug, _tgt, _conc = trn_dset[0]
    cell_dim, drug_dim = _cell.shape[0], _drug.shape[0]

    dataloader_kwargs = {
        'shuffle': 'True',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True}

    trn_loader = torch.utils.data.DataLoader(
        trn_dset, **dataloader_kwargs)
    tst_loaders = [torch.utils.data.DataLoader(
        _tst_dset, **dataloader_kwargs) for _tst_dset in tst_dsets]

    model = SimpleUno(cell_dim=cell_dim,
                      drug_dim=drug_dim,
                      state_dim=state_dim).to(device)

    optimizer = optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=4, min_lr=1e-6)

    def train():
        model.train()
        _trn_loss = 0.

        for _, cell, drug, trgt, dose in trn_loader:

            cell, drug, trgt, dose = cell.to(device), drug.to(device), \
                                     trgt.to(device), dose.to(device)
            optimizer.zero_grad()
            pred = model(cell, drug, dose)
            loss = F.mse_loss(pred, trgt)
            loss.backward()
            optimizer.step()

            _trn_loss += loss.item() * trgt.shape[0]

        return _trn_loss / len(trn_dset)

    def test():
        model.eval()

        tst_r2, tst_mae, tst_mse = [], [], []

        with torch.no_grad():
            for _tst_loader in tst_loaders:

                _tst_mse = 0.
                _tst_mae = 0.
                trgt_array = np.zeros(shape=(1,))
                pred_array = np.zeros(shape=(1,))

                for _, cell, drug, trgt, dose in _tst_loader:

                    cell, drug, trgt, concn = \
                        cell.to(device), drug.to(device), \
                        trgt.to(device), dose.to(device)

                    pred = model(cell, drug, concn)

                    _tst_mse += F.mse_loss(pred, trgt, reduction='sum')
                    _tst_mae += F.l1_loss(pred, trgt, reduction='sum')

                    trgt_array = np.concatenate(
                        (trgt_array, trgt.cpu().numpy().reshape(-1)))
                    pred_array = np.concatenate(
                        (pred_array, pred.cpu().numpy().reshape(-1)))

                tst_r2.append(r2_score(y_true=trgt_array, y_pred=pred_array))
                tst_mae.append(_tst_mae / len(_tst_loader.dataset))
                tst_mse.append(_tst_mse / len(_tst_loader.dataset))

        return tst_r2, tst_mae, tst_mse

    best_avg_r2 = float('-inf')
    best_epoch, early_stop_counter = 0, 0
    tst_history = []

    for epoch in range(1, 101):

        lr = scheduler.optimizer.param_groups[0]['lr']
        trn_loss = train()
        tst_r2, tst_mae, tst_mse = test()
        tst_history.append((tst_r2, tst_mae, tst_mse))

        print(f'Epoch {epoch:03d}, '
              f'LR = {lr:6f}, Training Loss = {trn_loss:.4f}.')
        for _i, _tst_source in enumerate(tst_sources):
            print(f'\tTest Results on {_tst_source}: '
                  f'R2 = {tst_r2[_i]:.4f}, '
                  f'MAE = {tst_mae[_i]:.4f}, '
                  f'MSE = {tst_mse[_i]:.4f}.')

        # Using average R2 score for learning rate adjustment and early stop
        scheduler.step(np.mean(tst_r2))
        if np.mean(tst_r2) > best_avg_r2:
            print(f'Best Avg R2: {np.mean(tst_r2)}')
            early_stop_counter = 0
            best_epoch = epoch
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print('No improvement on testing results. Stopping ... ')
                break
        print('-' * 80)

    print('#' * 80)
    print(f'Training Sources: {trn_sources} '
          f'(using only {subsample_percentage * 100: .0f}%% {subsample_on})')

    best_r2, best_mae, best_mse = tst_history[best_epoch - 1]
    print(f'Best Epoch {best_epoch}:')

    for _i, _tst_source in enumerate(tst_sources):
        print(f'\tTest Results on {_tst_source}: '
              f'R2 = {best_r2[_i]:.4f}, '
              f'MAE = {best_mae[_i]:.4f}, '
              f'MSE = {best_mse[_i]:.4f}.')

    print('#' * 80)
    print('#' * 80 + '\n')


def main():

    parser = argparse.ArgumentParser(description='Cross Study')

    parser.add_argument('--train_on', type=str, required=True, nargs='+',
                        choices=DATA_SOURCES)
    parser.add_argument('--test_on', type=str, required=True, nargs='+',
                        choices=DATA_SOURCES)

    parser.add_argument('--subsample_on', type=str, required=True,
                        choices=['cell', 'drug'])
    parser.add_argument('--lower_percentage', type=float, required=True)
    parser.add_argument('--higher_percentage', type=float, required=True)
    parser.add_argument('--percentage_increment', type=float, default=0.05)
    parser.add_argument('--state_dim', type=int, default=512)

    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()

    device = torch.device(f'cuda: {args.cuda_device}')
    seed_random_state(args.rand_state)

    subsample_percentage_array = np.arange(
        start=args.lower_percentage,
        step=args.percentage_increment,
        stop=args.higher_percentage + .01)

    for subsample_percentage in subsample_percentage_array:
        run_instance(trn_sources=args.train_on,
                     tst_sources=args.test_on,
                     state_dim=args.state_dim,
                     subsample_on=args.subsample_on,
                     subsample_percentage=subsample_percentage,
                     device=device)


if __name__ == '__main__':
    main()
