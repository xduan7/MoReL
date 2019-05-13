""" 
    File Name:          MoReL/pcba_similarity_matrix.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/5/19
    Python Version:     3.5.4
    File Description:   

"""
import numpy as np
from deepchem.molnet import load_pcba
from sklearn.decomposition import PCA

from featurizers import smiles_to_mols, mols_to_sim_mat, mols_to_ssm_mat, \
    FP_FUNC_DICT, SIM_FUNC_DICT


pcba_tasks, pcba_datasets, transformers = load_pcba(
    featurizer='Raw', split='scaffold', reload=True)
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

trn_smiles = train_dataset.ids
trn_target = train_dataset.y
print(f'Data Loaded.')

# len(trn_smiles) = 350,000
# which yields about 10 TB of features if each distances has 20 channels
# and stored in float32, and requires perhaps incremental PCA

# However, batch size of 32 will generate 0.896 GB of feature for iPCA
# Need to use smaller dataset and eliminates some of the not so important
# fingerprint distances

# Test out on a subset of molecules
subset_size = 4096
smiles_list = np.random.choice(trn_smiles, size=subset_size, replace=False)
mol_list = [mol for mol in smiles_to_mols(smiles_list) if mol is not None]
print(f'{len(mol_list)}/{subset_size} are valid molecules.')
subset_size = len(mol_list)

# Calculate the similarity matrix and substructure matching matrix
print(f'Computing similarity matrix of {subset_size} molecules ... ')
sim_mat = mols_to_sim_mat(mol_list,
                          fp_func_list=list(FP_FUNC_DICT.keys()),
                          sim_func_list=list(SIM_FUNC_DICT.keys()))
print(f'Computing substructure matching matrix '
      f'of {subset_size} molecules ... ')
ssm_mat = mols_to_ssm_mat(mol_list)

ssm_mat_pca = PCA(n_components=ssm_mat.shape[1] * ssm_mat.shape[2])
ssm_mat_pca.fit(ssm_mat.reshape(subset_size, -1))
print(ssm_mat_pca.explained_variance_ratio_)

sim_mat_pca = PCA(n_components=sim_mat.shape[1])
sim_mat_pca.fit(sim_mat.reshape(subset_size, -1))
print(sim_mat_pca.explained_variance_ratio_)
