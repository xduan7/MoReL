#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Local modules
import utils.data_prep.config as c
from utils.data_prep.download import download

download()


# In[12]:


# Prepare the CID-InChI hdf5
def inchi_prep(pcba_only: bool):
    
    # Check if the CID-InChI HDF5 file already exist ##########################
    cid_inchi_hdf5_path = c.PCBA_CID_INCHI_HDF5_PATH         if pcba_only else c.PC_CID_INCHI_HDF5_PATH
    
    if os.path.exists(cid_inchi_hdf5_path):
        msg = 'CID-InChI HDF5 already exits. '               'Press [Y] for overwrite, any other key to continue ... '
        if str(input(msg)).lower().strip() != 'y':
            return
        os.remove(cid_inchi_hdf5_path)
    
    cid_inchi_hdf5 = h5py.File(cid_inchi_hdf5_path, 'w', libver='latest')
    
    # Prepare the PCBA CID set if necessary ###################################
    pcba_cid_set = set(pd.read_csv(
        c.PCBA_CID_FILE_PATH,
        sep='\t',
        header=0,
        index_col=None,
        usecols=[0]).values.reshape(-1)) if pcba_only else set([])
    
    # Looping over all the CIDs ###############################################
    def __check_inchi(cid, inchi):
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return cid, inchi
        else:
            return cid, None
    
    for chunk_cid_inchi_df in pd.read_csv(c.CID_INCHI_FILE_PATH,
                                          sep='\t',
                                          header=None,
                                          index_col=[0],
                                          usecols=[0, 1],
                                          chunksize=2 ** 15):
    
        chunk_cid_inchi_list = Parallel(n_jobs=c.NUM_CORES)(
            delayed(__check_inchi)(cid, row[1])
            for cid, row in chunk_cid_inchi_df.iterrows()
            if ((not pcba_only) or (cid in pcba_cid_set)))
    
        for cid, row in chunk_cid_inchi_list:
            if row is not None:
                cid_inchi_hdf5.create_dataset(name=str(cid), data=row[1])
                
inchi_prep(pcba_only=True)


# In[ ]:


# Prepare for the CUD-Dragon7_descriptor hdf5
def d7_dscrptr_prep(target_only: bool):
    
    hdf5_path = c.PCBA_CID_TARGET_D7DSCPTR_HDF5_PATH if target_only         else c.PCBA_CID_D7DSCPTR_HDF5_PATH
    
    # Check if the descriptor file exists
    if os.path.exists(hdf5_path):
        msg = 'PCBA_CID-D7_Descriptor HDF5 already exits. '               'Press [Y] for overwrite, any other key to continue ... '
        if str(input(msg)).lower().strip() != 'y':
            return
        os.remove(hdf5_path)
    
    hdf5 = h5py.File(hdf5_path, 'w', libver='latest')

    # Load dragon7 descriptor targets #########################################
    read_csv_kwargs = {
        'sep':          '\t',
        'header':       0,
        'index_col':    0,
        'na_values':    'na',
        'dtype':        str,
        'chunksize':    2 ** 15
    }
    if target_only:
        read_csv_kwargs['usecols'] = ['NAME', ] + c.TARGET_D7_DSCRPTR_NAMES
    
    dscrptr_names_flag = True
    for chunk_pcba_cid_d7_dscrptr_df in         pd.read_csv(c.PCBA_CID_D7_DSCPTR_FILE_PATH, **read_csv_kwargs):
        
        if dscrptr_names_flag:
            dscrptr_names_flag = False
            hdf5.create_dataset(
                name='DSCRPTR_NAMES', 
                data=np.string_(list(chunk_pcba_cid_d7_dscrptr_df)))
        
        # Original dataset contains some NaN values
        chunk_pcba_cid_d7_dscrptr_df.dropna(inplace=True)
        
        for cid, row in chunk_pcba_cid_d7_dscrptr_df.iterrows():
            
            print(cid)
            print(row)
            
            hdf5.create_dataset(name=str(cid), 
                                data=np.array(row.values, dtype=np.float16))
        
        

d7_dscrptr_prep(target_only=True)

