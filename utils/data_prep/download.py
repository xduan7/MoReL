""" 
    File Name:          MoReL/download.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               2/11/19
    Python Version:     3.5.4
    File Description:   

"""
import os
import utils.data_prep.config as c


def create_dir():
    try:
        # os.mkdir(c.PROJECT_DIR)
        # os.mkdir(c.DATA_DIR)
        os.makedirs(c.RAW_DATA_DIR, exist_ok=True)
        # os.mkdir(c.PROCESSED_DATA_DIR)
        os.makedirs(c.CID_MOL_DATA_DIR, exist_ok=True)
        os.makedirs(c.CID_SMILES_DATA_DIR, exist_ok=True)
        os.makedirs(c.CID_ECFP_DATA_DIR, exist_ok=True)
        os.makedirs(c.CID_GRAPH_DATA_DIR, exist_ok=True)
    except FileExistsError:
        pass
    except Exception:
        print("Failed to create data directories.")
        raise


def download():
    """
    Download and unpack if raw data does not exist.
    """

    # Take care of all the data directories
    create_dir()

    # All the CID in PCBA dataset
    if not os.path.exists(c.PCBA_CID_FILE_PATH):
        os.system('wget -r -nd -nc %s -P %s'
                  % (c.PCBA_CID_FTP_ADDRESS, c.RAW_DATA_DIR))
        os.system('find %s -type f -iname \"*.gz\" -exec gunzip {} +' %
                  c.RAW_DATA_DIR)

    # All the CID-InChI one-on-one lookup
    if not os.path.exists(c.CID_INCHI_FILE_PATH):
        os.system('wget -r -nd -nc %s -P %s'
                  % (c.CID_INCHI_FTP_ADDRESS, c.RAW_DATA_DIR))
        os.system('find %s -type f -iname \"*.gz\" -exec gunzip {} +'
                  % c.RAW_DATA_DIR)
