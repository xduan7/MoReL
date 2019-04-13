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

# This boolean controls whether to count the atoms during data preparation
# Counting atoms means that we are going to check the occurrence of all the
# atoms in each molecule, and compute the overall occurrence of a specific
# atom in all the molecules. Note that each type of atom will only count
# once even if it appears in a molecule multiple times.
COUNTING_ATOMS = False

# Parallelization of data prep
# (#cpu-2) to prevent system crush/no response
NUM_CORES = min(multiprocessing.cpu_count() - 1, 8)

# Indicator for using a subset (PCBA)
# The full set contains about 135 million molecules
# The subset (PCBA) contains about 3 million molecules
PCBA_ONLY = True
DATASET_INDICATOR = '(PCBA)' if PCBA_ONLY else '(full)'

# Number of validation samples if it is int; else it's validation ratio
VALIDATION_SIZE = 5000

# Encoding method from molecule binary to string
# Note that 'base64' is more spatially efficient than 'hex'
MOL_BINARY_ENCODING = 'base64'

# Molecule processing specifications
# The explicit display of all bonds in a SMILES string, for example
# * set to False: 'CC(=O)OC(CC(=O)O)C[N+](C)(C)C'
# * set to True: 'C-C(=O)-O-C(-C-C(=O)-O)-C-[N+](-C)(-C)-C'
ALL_BONDS_EXPLICIT = False

# The explicit display of H atoms in a SMILES string, for example
# * set to False: 'C1CCCCC1'
# * set to True: '[CH2]1[CH2][CH2][CH2][CH2][CH2]1'
ALL_HS_EXPLICIT = False


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
# CID_MOL_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-Mol/')
# CID_SMILES_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-SMILES/')
# CID_ECFP_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-ECFP/')
# CID_GRAPH_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-Graph/')

# Raw files and FTP locations #################################################

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
# Processed files locations ###################################################

PC_CID_INCHI_HDF5_PATH = join(PROCESSED_DATA_DIR, 'CID-InChI.hdf5')
PCBA_CID_INCHI_HDF5_PATH = join(PROCESSED_DATA_DIR, 'CID-InChI(PCBA).hdf5')
PC_CID_SMILES_CSV_PATH = join(PROCESSED_DATA_DIR, 'CID-SMILES.csv')
PCBA_CID_SMILES_CSV_PATH = join(PROCESSED_DATA_DIR, 'CID-SMILES(PCBA).csv')

CID_MOL_STR_HDF5_PATH = join(PROCESSED_DATA_DIR,
                             'CID-Mol_str%s.hdf5' % DATASET_INDICATOR)

# Combined HDF5 for all features (different groups)
CID_FEATURES_HDF5_PATH = join(
    PROCESSED_DATA_DIR, 'CID-features%s.hdf5' % DATASET_INDICATOR)

# Atom dictionary (atom symbol - occurrence / num_compounds)
ATOM_DICT_TXT_PATH = join(PROCESSED_DATA_DIR,
                          'atom_dict%s.txt' % DATASET_INDICATOR)
# Unused CID
# A compound could be eliminated from the dataset for the following reasons:
# * failed to construct Chem.Mol object from InChI
# * too many atoms ( > MAX_NUM_ATOMS);
# * too many characters in its SMILES string ( > MAX_LEN_SMILES);
UNUSED_CID_TXT_PATH = join(PROCESSED_DATA_DIR,
                           'unused_CID%s.txt' % DATASET_INDICATOR)


# # SMILES and tokenized feature
# CID_SMILES_HDF5_PATH = join(
#     PROCESSED_DATA_DIR, 'CID-SMILES%s.hdf5' % DATASET_INDICATOR)
# CID_TOKEN_HDF5_PATH = join(
#     PROCESSED_DATA_DIR, 'CID-token%s.hdf5' % DATASET_INDICATOR)
#
# # ECFP in either array format or Base64 encoding
# CID_ECFP_HDF5_PATH = join(
#     PROCESSED_DATA_DIR, 'CID-ECFP%s.hdf5' % DATASET_INDICATOR)
# CID_BASE64_ECFP_HDF5_PATH = join(
#     PROCESSED_DATA_DIR, 'CID-ECFP_base64%s.hdf5' % DATASET_INDICATOR)
#
# # Graph features: dense matrices of nodes and edge features
# CID_GRAPH_HDF5_PATH = join(
#     PROCESSED_DATA_DIR, 'CID-Graph%s.hdf5' % DATASET_INDICATOR)

PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH = join(PROCESSED_DATA_DIR,
                                          'PCBA_CID-d7descriptor(target).hdf5')
PCBA_CID_D7DSCPTR_HDF5_PATH = join(PROCESSED_DATA_DIR,
                                   'PCBA_CID-d7descriptor.hdf5')

PCBA_CID_TARGET_D7DSCPTR_CSV_PATH = join(PROCESSED_DATA_DIR,
                                         'CID-target_DD(PCBA).csv')
PCBA_CID_D7DSCPTR_CSV_PATH = join(PROCESSED_DATA_DIR,
                                  'CID-DD(PCBA).csv')

# Models and training constants ###############################################
FEATURES = []
MODELS = {
    'dense':    ['tokens'],
    'xfmr':     ['tokens', ],
    'ggnn':     ['graph', ],
}

# Test size: the number of molecules used for testing
# For PCBA dataset, there are about 3 million molecules
# For the whole PubChem dataset, the size is about 94 million
TEST_SIZE = 65536

# The timeout in ms for shared dict structure
# Features that are inserted [] ms ago or earlier will be removed
SHARED_DICT_TIMEOUT_MS = 4096

# Maximum byte size for mmap file. Set to 16 MB, which should be sufficient
# for 16384 features of size 1024 with dtype of np.uint8
MMAP_BYTE_SIZE = 2 ** 24

# Number of PyTorch dataloader workers
# Set to 0 to simulate CPU cluster environment
NUM_DATALOADER_WORKERS = 0

#
SHARED_DICT_TIMEOUT_BATCH = 16




