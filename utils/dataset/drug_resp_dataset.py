""" 
    File Name:          MoReL/drug_resp_dataset.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/22/19
    Python Version:     3.5.4
    File Description:   

"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
import utils.data_prep.config as c


# Suppress unnecessary RDkit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def get_drug_resp_array(target: str):
    pd.read_csv(c.DRUG_RESP_FILE_PATH)
    return


def get_drug_dict(drug_list: list = None):

    drug_info_array = pd.read_csv(
        c.DRUG_INFO_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=['ID', 'SMILES', 'INCHIKEY']).values

    drug_dict = {}
    for row in drug_info_array:

        id, smiles, inchi = str(row[0]), str(row[1]), str(row[2])
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.MolFromInchi(inchi) if mol is None else mol

        if (mol is None) or \
                ((drug_list is not None) and (id not in drug_list)):
            continue

        drug_dict[id] = Chem.MolToSmiles(mol)

    return drug_dict


def get_cell_dict(cell_list: list):
    pd.read_csv(c.RNA_SEQ_FILE_PATH)
    return


def trn_tst_split(on_drug: bool, on_cell: bool):
    return


class DrugRespDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


# Testing segment
if __name__ == '__main__':

    # Load drug response data, drug dict, and cell dict
    drug_list = []
    cell_list = []

    # Performing training/testing split based on cell, drug, or combined
