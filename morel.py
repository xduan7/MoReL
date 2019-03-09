""" 
    File Name:          MoReL/morel.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               3/9/19
    Python Version:     3.5.4
    File Description:   

"""
import mmap
import pickle
from argparse import Namespace
from multiprocessing import Manager, Process, Lock

import utils.data_prep.config as c
from morel_instance import start_instance

if __name__ == '__main__':

    NUM_DEVICES = 2

    base_args_dict = {
        # 'process_id': 0,
        'rand_state': 0,
        # 'device': 'cuda:0',

        # Dataloader parameters
        'feature_type': 'ecfp',
        # 'featurization': 'computing',
        'dict_timeout_ms': c.SHARED_DICT_TIMEOUT_MS,
        'target_dscrptr_name': 'CIC5',

        # Model parameters
        'model_type': 'dense',
        'dense_num_layers': 4,
        'dense_feature_dim': 2048,
        'dense_emb_dim': 4096,
        # 'dense_dropout': 0.2,

        # Optimizer and other parameters
        'train_batch_size': 32,
        'test_batch_size': 2048,
        'max_num_epochs': 1,
        'max_batches_per_epoch': 1024,
        'optimizer': 'sgd',
        'learing_rate': 1e-3,
        'l2_regularization': 1e-5,

        # Debug
        'debug': False,
    }

    for featurization in ['computing', 'mmap', 'dict_proxy']:

        print('#' * 80)
        print('Getting features with %s method ... ' % featurization)

        if featurization == 'mmap':
            shared_dict = mmap.mmap(fileno=-1,
                                    length=c.MMAP_BYTE_SIZE,
                                    access=mmap.ACCESS_WRITE)
            shared_dict.seek(0)
            shared_dict.write(pickle.dumps({}))
        elif featurization == 'dict_proxy':
            manager = Manager()
            shared_dict = manager.dict()
        else:
            shared_dict = None

        shared_lock = Lock()

        process = {}
        for id in range(NUM_DEVICES):

            proc_args_dict = base_args_dict.copy()
            proc_args_dict['process_id'] = id
            proc_args_dict['device'] = 'cuda:%i' % id
            proc_args_dict['featurization'] = featurization

            # Different network for each process?
            proc_args_dict['dense_dropout'] = id / 10.
            # proc_args_dict['dense_num_layers'] += id
            # proc_args_dict['l2_regularization'] *= id

            proc_args = Namespace(**proc_args_dict)

            process[id] = Process(target=start_instance,
                                  args=(proc_args, shared_dict, shared_lock))
            process[id].start()

        for id in range(NUM_DEVICES):
            process[id].join()
