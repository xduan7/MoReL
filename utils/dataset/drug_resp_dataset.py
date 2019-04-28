""" 
    File Name:          MoReL/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/22/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

import utils.data_prep.config as c

# Suppress unnecessary RDkit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def select_resp_array(resp_array,
                      cell_list: list = None,
                      drug_list: list = None):

    if (drug_list is None) and (cell_list is None):
        return resp_array
    else:
        cell_set = set(cell_list)
        drug_set = set(drug_list)

        resp_list = []
        for row in resp_array:
            cell_id, drug_id = row[1], row[2]

            if (cell_set is not None) and (cell_id not in cell_set):
                continue

            if (drug_set is not None) and (drug_id not in drug_set):
                continue

            resp_list.append(row)
        return np.array(resp_list)


def get_resp_array(target: str = 'AUC',
                   cell_list: list = None,
                   drug_list: list = None):

    resp_array = pd.read_csv(c.DRUG_RESP_FILE_PATH,
                             sep='\t', header=0, index_col=None,
                             usecols=['SOURCE', 'CELL', 'DRUG', target]).values

    # Change the dtype of prediction target
    resp_array[:, 3] = np.array(resp_array[:, 3], dtype=np.float32)

    return select_resp_array(resp_array,
                             cell_list=cell_list,
                             drug_list=drug_list)


def get_cell_dict(cell_list: list = None):
    cell_array = pd.read_csv(c.RNA_SEQ_FILE_PATH,
                             sep='\t', header=0, index_col=None).values

    cell_dict = {}
    for row in cell_array:

        cell_id = str(row[0])
        seq = np.array(row[1:], dtype=np.float32)

        if (cell_list is not None) and (cell_id not in cell_list):
            continue

        cell_dict[cell_id] = seq

    return cell_dict


def get_drug_dict(drug_list: list = None):
    drug_array = pd.read_csv(
        c.DRUG_SMILES_FILE_PATH,
        sep='\t',
        header=None,
        index_col=None).values

    drug_dict = {}
    for row in drug_array:

        smiles, drug_id = row[0], row[1]
        mol = Chem.MolFromSmiles(smiles)

        if (mol is None) or \
                ((drug_list is not None) and (drug_id not in drug_list)):
            print(f'Failed to transfer {drug_id} with '
                  f'SMILES string {smiles} into Chem.Mol object')
            continue

        drug_dict[drug_id] = Chem.MolToSmiles(mol)

    return drug_dict


def trn_tst_split(resp_array: np.array,
                  test_ratio: float = 0.2,
                  rand_state: int = 0,
                  disjoint_cells: bool = True,
                  disjoint_drugs: bool = False):

    # If drugs and cells are not specified to be disjoint in the training
    # and testing dataset, then random split stratified on data sources
    if (not disjoint_cells) and (not disjoint_drugs):
        __source_array = resp_array[:, 0]
        return train_test_split(resp_array,
                                test_size=test_ratio,
                                random_state=rand_state,
                                stratify=__source_array)

    __cell_array = np.unique(resp_array[:, 1])
    __drug_array = np.unique(resp_array[:, 2])

    # Adjust the split ratio if both cells and drugs are disjoint
    # Note that mathematically speaking, we should adjust
    __test_ratio = test_ratio ** 0.7 if (disjoint_cells and disjoint_drugs) \
        else test_ratio

    if disjoint_cells:
        __trn_cell_array, __tst_cell_array = \
            train_test_split(__cell_array,
                             test_size=__test_ratio,
                             random_state=rand_state)
    else:
        __trn_cell_array, __tst_cell_array = __cell_array, __cell_array

    if disjoint_drugs:
        __trn_drug_array, __tst_drug_array = \
            train_test_split(__drug_array,
                             test_size=__test_ratio,
                             random_state=rand_state)
    else:
        __trn_drug_array, __tst_drug_array = __drug_array, __drug_array

    __trn_resp_array = select_resp_array(resp_array,
                                         cell_list=list(__trn_cell_array),
                                         drug_list=list(__trn_drug_array))
    __tst_resp_array = select_resp_array(resp_array,
                                         cell_list=list(__tst_cell_array),
                                         drug_list=list(__tst_drug_array))
    return __trn_resp_array, __tst_resp_array


class DrugRespDataset(Dataset):

    def __init__(self,
                 cell_dict: dict,
                 drug_dict: dict,
                 resp_array: np.array,
                 featurizer: callable,
                 featurizer_kwargs: dict):

        super().__init__()

        self.__cell_dict = cell_dict
        self.__drug_dict = drug_dict
        self.__resp_array = resp_array

        self.__featurizer = featurizer
        self.__featurizer_kwargs = featurizer_kwargs

        self.__sources = c.DRUG_RESP_SOURCES
        self.__len = len(self.__resp_array)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):

        resp_data = self.__resp_array[index]
        source, cell_id, drug_id, target = \
            resp_data[0], resp_data[1], resp_data[2], resp_data[3]

        source_data = np.zeros_like(self.__sources, dtype=np.float32)
        source_data[self.__sources.index(source)] = 1.
        source_data = torch.from_numpy(source_data)

        cell_data = torch.from_numpy(self.__cell_dict[cell_id])

        drug_smiles = self.__drug_dict[drug_id]
        drug_mol = Chem. MolFromSmiles(drug_smiles)
        drug_data = self.__featurizer(
            mol=drug_mol, **self.__featurizer_kwargs)
        if self.__featurizer == mol_to_graph:
            n, adj, e = drug_data
            drug_data = Data(x=torch.from_numpy(n),
                             edge_index=torch.from_numpy(adj),
                             edge_attr=torch.from_numpy(e))
        else:
            drug_data = torch.from_numpy(drug_data)

        target_data = torch.from_numpy(np.array([target, ], dtype=np.float32))

        return source_data, cell_data, drug_data, target_data


# Testing segment
if __name__ == '__main__':

    from utils.data_prep.featurizers import mol_to_graph

    # This is an example of dataloading for drug response dataset
    # Load drug response data, drug dict, and cell dict
    cell_dict = get_cell_dict()
    drug_dict = get_drug_dict()

    # This step made sure that all the drugs and cells are present in the dicts
    resp_array = get_resp_array(cell_list=list(cell_dict.keys()),
                                drug_list=list(drug_dict.keys()))

    # Testing out the training/testing split
    for dc in [False, True]:
        for dd in [False, True]:

            print(f'Splitting data with disjoint_cells = {dc}'
                  f' and disjoint_drugs = {dd}')

            trn_data, tst_data = trn_tst_split(resp_array,
                                               disjoint_cells=dc,
                                               disjoint_drugs=dd)
            len_data = (len(trn_data) + len(tst_data))

            print(f'Training/testing ratio = '
                  f'{len(tst_data) / len_data: .3f}; \t'
                  f'Data usage = {len_data / len(resp_array): .3f}')

    mol_to_graph_kwargs = {
        'master_atom': True,
        'master_bond': True,
        'max_num_atoms': -1}

    dset = DrugRespDataset(cell_dict=cell_dict,
                           drug_dict=drug_dict,
                           resp_array=resp_array,
                           featurizer=mol_to_graph,
                           featurizer_kwargs=mol_to_graph_kwargs)


