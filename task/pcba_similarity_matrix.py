""" 
    File Name:          MoReL/pcba_similarity_matrix.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/5/19
    Python Version:     3.5.4
    File Description:   

"""
from deepchem.molnet import load_pcba


pcba_tasks, pcba_datasets, transformers = load_pcba(
    featurizer='Raw', split='random', reload=True)
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

trn_smiles = train_dataset.ids
trn_target = train_dataset.y

# len(trn_smiles) = 350,000
# which yields about 10 TB of features if each distances has 20 channels
# and stored in float32, and requires perhaps incremental PCA

# However, batch size of 32 will generate 0.896 GB of feature for iPCA
# Need to use smaller dataset and eliminates some of the not so important
# fingerprint distances


