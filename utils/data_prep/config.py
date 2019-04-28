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
import multiprocessing
from rdkit import Chem, RDLogger
from os.path import abspath, join

# Suppress unnecessary RDkit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# File processing and data preparation configurations #########################

RANDOM_STATE = 0

# Parallelization of data prep
# (#cpu-2) to prevent system crush/no response
NUM_CORES = min(multiprocessing.cpu_count() - 1, 8)

# Indicator for using a subset (PCBA)
# The full set contains about 135 million molecules
# The subset (PCBA) contains about 3 million molecules
PCBA_ONLY = True
DATASET_INDICATOR = '(PCBA)' if PCBA_ONLY else '(full)'

# Dragon7 descriptor (target) names ###########################################
TARGET_D7_DSCRPTR_NAMES = [
    'MW', 'AMW', 'VE1sign_B(s)', 'IC1', 'GATS4e', 'RBF', 'GATS3e', 'Eta_sh_y',
    'Mp', 'GATS5i', 'IC2', 'MATS7e', 'Mi', 'Mv', 'Me', 'SssO', 'VE3sign_B(s)',
    'MATS5i', 'GATS3s', 'MATS7i', 'MATS3e', 'GATS5m', 'MATS5m', 'PW2',
    'GATS6e', 'MATS4m', 'MATS6s', 'GATS1s', 'SpPosA_X', 'GATS5s', 'P_VSA_MR_5',
    'VE1sign_D/Dt', 'ALOGP', 'BIC2', 'MATS3v', 'VE1sign_D', 'MATS5s', 'MATS3s',
    'VE1sign_Dt', 'VE3sign_D', 'MATS2e', 'SIC2', 'GATS8s', 'MATS4p', 'GATS8e',
    'VE3sign_Dt', 'GATS4m', 'O%', 'MATS8p', 'GATS5e', 'MATS3i', 'PJI2', 'BIC4',
    'CATS2D_02_AL', 'MATS8i', 'MATS3p', 'GATS7m', 'GATS6m', 'JGT',
    'VE1sign_Dz(p)', 'MATS4e', 'N%', 'MATS1e', 'P_VSA_LogP_4', 'GATS2i', 'PW3',
    'MATS4s', 'VE1_B(s)', 'P_VSA_LogP_3', 'MATS4i', 'MATS2s', 'JGI2', 'MATS6m',
    'MATS7p', 'GATS8p', 'GATS8v', 'GATS4s', 'GATS5p', 'TI2_L', 'MATS6p',
    'MATS5e', 'GATS4i', 'VE3sign_D/Dt', 'GATS3m', 'VE1sign_X', 'P_VSA_LogP_2',
    'MAXDN', 'nBM', 'MATS8m', 'MATS1s', 'GATS4p', 'GATS1e', 'D/Dtr05',
    'MATS3m', 'BLI', 'P_VSA_MR_7', 'GATS7s', 'MATS6i', 'MATS5p', 'MATS7m']
# Missing 'CIC5', 'SpDiam_B(m)', 'VE1_A', 'SM6_H2', 'SM14_AEA(dm)',
# 'SpMin1_Bh(e)', 'SpMax1_Bh(s)'

# Directories #################################################################
PROJECT_DIR = abspath(join(abspath(__file__), '../../../'))
DATA_DIR = join(PROJECT_DIR, 'data/')
RAW_DATA_DIR = join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = join(DATA_DIR, 'processed/')


# Drug response data ##########################################################
DRUG_RESP_SOURCES  = ['CCLE', 'CTRP', 'GDSC', 'NCI60', 'gCSI']

DRUG_INFO_FILE_NAME = 'drug_info'
DRUG_SMILES_FILE_NAME = 'pan_drugs_combined.smiles'
RNA_SEQ_FILE_NAME = 'combined_rnaseq_data_lincs1000'
DRUG_RESP_FILE_NAME = 'combined_single_response_agg'

DRUG_INFO_FILE_PATH = join(RAW_DATA_DIR, DRUG_INFO_FILE_NAME)
DRUG_SMILES_FILE_PATH = join(RAW_DATA_DIR, DRUG_SMILES_FILE_NAME)
RNA_SEQ_FILE_PATH = join(RAW_DATA_DIR, RNA_SEQ_FILE_NAME)
DRUG_RESP_FILE_PATH = join(RAW_DATA_DIR, DRUG_RESP_FILE_NAME)

# PubChem molecule representation data ########################################
# CID in PCBA dataset, ranging from 2 to 135,693,611, total size ~3,400,000
PCBA_CID_FILE_NAME = 'Cid2BioactivityLink'
CID_INCHI_FILE_NAME = 'CID-InChI-Key'
PCBA_CID_D7_DSCPTR_FILE_NAME = 'PCBA-CID_dragon7_descriptors.tsv'
PCBA_CID_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/%s.gz' \
    % PCBA_CID_FILE_NAME
CID_INCHI_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/%s.gz' \
    % CID_INCHI_FILE_NAME
PCBA_CID_D7_DSCPTR_ADDRESS = \
    'http://bioseed.mcs.anl.gov/~fangfang/PCBA-CID/%s' \
    % PCBA_CID_D7_DSCPTR_FILE_NAME
PCBA_CID_FILE_PATH = join(RAW_DATA_DIR, PCBA_CID_FILE_NAME)
CID_INCHI_FILE_PATH = join(RAW_DATA_DIR, CID_INCHI_FILE_NAME)
PCBA_CID_D7_DSCPTR_FILE_PATH = join(RAW_DATA_DIR, PCBA_CID_D7_DSCPTR_FILE_NAME)

PC_CID_SMILES_CSV_PATH = join(PROCESSED_DATA_DIR, 'CID-SMILES.csv')
PCBA_CID_SMILES_CSV_PATH = join(PROCESSED_DATA_DIR, 'CID-SMILES(PCBA).csv')
PCBA_CID_TARGET_D7DSCPTR_CSV_PATH = \
    join(PROCESSED_DATA_DIR, 'CID-target_DD(PCBA).csv')
PCBA_CID_D7DSCPTR_CSV_PATH = \
    join(PROCESSED_DATA_DIR, 'CID-DD(PCBA).csv')
