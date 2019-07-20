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
from utils.dataset.drug_resp_dataset import DrugRespDataset, trim_resp_array, \
    get_resp_array, ScalingMethod, NanProcessing, DrugFeatureType, \
    CellProcessingMethod, CellSubsetType, CellDataType, get_datasets, \
    SubsampleType


# Simple Uno-like model
class SimpleUno(nn.Module):

    def __init__(self,
                 cell_dim: int,
                 drug_dim: int,
                 state_dim: int = 512,
                 dropout: float = 0.2):

        super(SimpleUno, self).__init__()

        self.__cell_tower = nn.Sequential(
            nn.Linear(cell_dim, state_dim, bias=True),
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
            nn.Dropout(dropout))

        self.__drug_tower = nn.Sequential(
            nn.Linear(drug_dim, state_dim, bias=True),
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

            nn.Linear(state_dim, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1, bias=True),
            nn.ReLU())

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
        resp_data_path='/raid/xduan7/Data/combined_single_drug_growth.txt',
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

    tmp_resp_array = get_resp_array(
        data_path='/raid/xduan7/Data/combined_single_drug_growth.txt',
        aggregated=False,
        target='GROWTH',
        data_sources=tst_sources)
    tmp_resp_array = trim_resp_array(
        resp_array=tmp_resp_array,
        cells=trn_cell_dict.keys(),
        drugs=trn_drug_dict.keys(),
        inclusive=True)
    tst_dset = DrugRespDataset(
        cell_dict=trn_cell_dict,
        drug_dict=trn_drug_dict,
        resp_array=tmp_resp_array,
        aggregated=False)

    # Subsample the training set either on drug or cell
    subsample_type = SubsampleType(subsample_on)
    trn_dset.subsample(subsample_type, subsample_percentage)

    return trn_dset, tst_dset


def run_instance(
        trn_sources: List[str],
        tst_sources: List[str],
        subsample_on: str,
        subsample_percentage: float,
        device: torch.device):

    print('\n' + '#' * 80)
    print('#' * 80)

    print(f'Training Sources: {trn_sources} '
          f'(using only {subsample_percentage * 100: .0f}%% {subsample_on})')

    trn_dset, tst_dset = \
        get_cross_study_datasets(trn_sources=trn_sources,
                                 tst_sources=tst_sources,
                                 subsample_on=subsample_on,
                                 subsample_percentage=subsample_percentage)

    print('Datasets Summary:')
    print(trn_dset)
    print(tst_dset)
    print('#' * 80)

    # Get the dimensions of features in the most awkward way possible
    _src, _cell, _drug, _tgt, _conc = trn_dset[0]
    cell_dim, drug_dim = _cell.shape[0], _drug.shape[0]

    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        'batch_size': 32,
        'num_workers': 8,
        'pin_memory': False}

    trn_loader = torch.utils.data.DataLoader(
        trn_dset, **dataloader_kwargs)
    tst_loader = torch.utils.data.DataLoader(
        tst_dset, **dataloader_kwargs)

    model = SimpleUno(cell_dim=cell_dim,
                      drug_dim=drug_dim).to(device)

    optimizer = optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=8, min_lr=1e-5)

    def train():
        model.train()
        _trn_loss = 0.

        for _, cell, drug, trgt, concn in trn_loader:

            cell, drug, trgt, concn = cell.to(device), drug.to(device), \
                                      trgt.to(device), concn.to(device)
            optimizer.zero_grad()
            pred = model(cell, drug, concn)
            loss = F.mse_loss(pred, trgt)
            loss.backward()
            optimizer.step()

            _trn_loss += loss.item() * trgt.shape[0]

        return _trn_loss / len(trn_dset)

    def test():
        model.eval()
        _tst_mse = 0.
        _tst_mae = 0.
        trgt_array = np.zeros(shape=(1, ))
        pred_array = np.zeros(shape=(1, ))

        with torch.no_grad():
            for _, cell, drug, trgt, concn in tst_loader:

                cell, drug, trgt, concn = cell.to(device), drug.to(device), \
                                          trgt.to(device), concn.to(device)
                pred = model(cell, drug, concn)

                _tst_mse += F.mse_loss(pred, trgt, reduction='sum')
                _tst_mae += F.l1_loss(pred, trgt, reduction='sum')

                trgt_array = np.concatenate(
                    (trgt_array, trgt.cpu().numpy().reshape(-1)))
                pred_array = np.concatenate(
                    (pred_array, pred.cpu().numpy().reshape(-1)))

        _tst_r2 = r2_score(y_true=trgt_array, y_pred=pred_array)

        return _tst_r2, _tst_mae / len(tst_dset), _tst_mse / len(tst_dset)

    best_r2 = float('-inf')
    best_mae, best_mse = None, None
    for epoch in range(1, 501):

        lr = scheduler.optimizer.param_groups[0]['lr']
        trn_loss = train()
        tst_r2, tst_mae, tst_mse = test()
        scheduler.step(tst_r2)

        print(f'Epoch: {epoch:03d}, '
              f'LR: {lr:6f}, Training Loss: {trn_loss:.4f}.\n',
              f'Testing Results:\n'
              f'\tR2: {tst_r2:.4f} MAE: {tst_mae:.4f} MSE: {tst_mse:.4f}.')

        if tst_r2 > best_r2:
            best_r2, best_mae, best_mse = tst_r2, tst_mae, tst_mse

    print('#' * 80)
    print(f'Training Sources: {trn_sources} '
          f'(using only {subsample_percentage * 100: .0f}%% {subsample_on})')
    print(f'Testing Sources: {tst_sources}')
    print(f'Best R2 {best_r2:.4f} '
          f'(MAE = {best_mae:.4f}, MSE = {best_mse:.4f})')

    print('#' * 80)
    print('#' * 80 + '\n')


def main():

    parser = argparse.ArgumentParser(description='Cross Study')

    parser.add_argument('--subsample_on', type=str, required=True,
                        choices=['cell', 'drug'])
    parser.add_argument('--lower_percentage', type=float, required=True)
    parser.add_argument('--higher_percentage', type=float, required=True)
    parser.add_argument('--percentage_increment', type=float, default=0.05)

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
        stop=args.higher_percentage + .1)

    for subsample_percentage in subsample_percentage_array:
        run_instance(trn_sources=['CTRP', ],
                     tst_sources=['GDSC', ],
                     subsample_on=args.subsample_on,
                     subsample_percentage=subsample_percentage,
                     device=device)


if __name__ == '__main__':
    main()
