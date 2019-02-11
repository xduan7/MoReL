""" 
    File Name:          MoReL/config.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

        This file saves all the constants that are related to data
        downloading, pre-processing and storing.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.realpath(__file__)))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed/')
CID_MOL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'CID-Mol/')
CID_SMILES_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'CID-SMILES/')
CID_ECFP_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'CID-ECFP/')
CID_GRAPH_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'CID-Graph/')


PCBA_CID_FILE_NAME = 'Cid2BioactivityLink'
CID_INCHI_FILE_NAME = 'CID-InChI-Key'


PCBA_CID_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/%s.gz' \
    % PCBA_CID_FILE_NAME
CID_INCHI_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/%s.gz' \
    % CID_INCHI_FILE_NAME


# CID in PCBA dataset, ranging from 2 to 135,693,611, total size ~3,400,000
PCBA_CID_FILE_PATH = os.path.join(RAW_DATA_DIR, PCBA_CID_FILE_NAME)
CID_INCHI_FILE_PATH = os.path.join(RAW_DATA_DIR, CID_INCHI_FILE_NAME)


MAX_NUM_ATOM = 32
MAX_SMILES_LEN = 128

RANDOM_STATE = 0
VALIDATION_SIZE = 5000


UNUSED_CID_FILE_NAME = 'unused_CID.txt'
UNUSED_CID_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, UNUSED_CID_FILE_NAME)
CID_MOL_FILE_NAME = 'CID-Mol.hdf5'
CID_MOL_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, CID_MOL_FILE_NAME)

# Need to specify the FP (probably using ECFP4(1024) + ECFP6(1024))
# CID_ECFP_FILE_NAME
# CID_ECFP_FILE_PATH

# Need to decide either using sparse matrix of dense one
# CID_GRAPH_FILE_NAME
# CID_GRAPH_FILE_NAME

CHUNK_CID_FILE_NAME = './CID-Mol/chunk_num-CID.txt'
CHUNK_CID_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, CHUNK_CID_FILE_NAME)




