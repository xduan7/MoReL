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
from os.path import abspath, join


# File processing and data preparation configurations #########################

RANDOM_STATE = 0

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

# Three requirements for the molecules
MAX_NUM_ATOMS = 64
MAX_LEN_SMILES = 128

# This is the size of stored string length in HDF5
# Note that 1024 is sufficient for molecules with less than 65 atoms mostly
MAX_LEN_MOL_STR = 1024

# Directories #################################################################

PROJECT_DIR = abspath(join(abspath(__file__), '../../../'))
DATA_DIR = join(PROJECT_DIR, 'data/')
RAW_DATA_DIR = join(DATA_DIR, 'raw/')
PROCESSED_DATA_DIR = join(DATA_DIR, 'processed/')
CID_MOL_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-Mol/')
CID_SMILES_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-SMILES/')
CID_ECFP_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-ECFP/')
CID_GRAPH_DATA_DIR = join(PROCESSED_DATA_DIR, 'CID-Graph/')

# Raw files and FTP locations #################################################

# CID in PCBA dataset, ranging from 2 to 135,693,611, total size ~3,400,000
PCBA_CID_FILE_NAME = 'Cid2BioactivityLink'
CID_INCHI_FILE_NAME = 'CID-InChI-Key'
PCBA_CID_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/%s.gz' \
    % PCBA_CID_FILE_NAME
CID_INCHI_FTP_ADDRESS = \
    'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/%s.gz' \
    % CID_INCHI_FILE_NAME
PCBA_CID_FILE_PATH = join(RAW_DATA_DIR, PCBA_CID_FILE_NAME)
CID_INCHI_FILE_PATH = join(RAW_DATA_DIR, CID_INCHI_FILE_NAME)

# Processed files locations ###################################################

# Atom dictionary (atom symbol - occurrence / num_compounds)
ATOM_DICT_FILE_PATH = join(PROCESSED_DATA_DIR,
                           'atom_dict%s.txt' % DATASET_INDICATOR)
# Unused CID
# A compound could be eliminated from the dataset for the following reasons:
# * failed to construct Chem.Mol object from InChI
# * too many atoms ( > MAX_NUM_ATOMS);
# * too many characters in its SMILES string ( > MAX_LEN_SMILES);
# * too many characters in the string representation of binary of its Chem.Mol
#       object ( > MAX_LEN_MOL_STR)
UNUSED_CID_FILE_PATH = join(PROCESSED_DATA_DIR,
                            'unused_CID%s.txt' % DATASET_INDICATOR)
CID_MOL_FILE_PATH = join(PROCESSED_DATA_DIR,
                         'CID-Mol%s.hdf5' % DATASET_INDICATOR)

# Need to specify the FP (probably using ECFP4(1024) + ECFP6(1024))
# CID_ECFP_FILE_PATH

# Need to decide either using sparse matrix of dense one
# CID_GRAPH_FILE_PATH

