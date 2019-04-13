""" 
    File Name:          MoReL/morel_instance.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               3/3/19
    Python Version:     3.5.4
    File Description:   
        This file contains the training, testing, and the looping function
        for a MoReL instance.
"""
import json
import pickle
import random
import time
import numpy as np
from multiprocessing import Manager, Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import mmap
from multiprocessing.managers import DictProxy

from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

import utils.data_prep.config as c
from network.combo_model import ComboModel
from utils.dataset.combo_dataset import ComboDataset
from utils.misc.optimizer import get_optimizer
from utils.misc.random_seeding import seed_random_state


def train(args: Namespace,
          model: nn.Module,
          optim: Optimizer,
          dataloader: DataLoader):
    start_ms = int(round(time.time() * 1000))
    model.train()
    for batch_index, (feature, target) in enumerate(dataloader):

        feature, target = \
            feature.to(args.device, dtype=torch.float32), \
            target.to(args.device, dtype=torch.float32)

        optim.zero_grad()
        prediction = model(feature)
        loss = F.mse_loss(input=prediction, target=target)
        loss.backward()
        optim.step(closure=None)

        # gpu_util = 100 * (1 - (dataloader.dataset.getitem_time_ms
        #                        / (int(round(time.time() * 1000))
        #                        - start_ms)))
        # print('Batch %i \t Loss = %f. GPU utilization %.2f%%'
        #       % (batch_index, loss.item(), gpu_util))

        if (batch_index + 1) >= args.max_batches_per_epoch:
            break

    overall_time_ms = (int(round(time.time() * 1000)) - start_ms)
    dataloading_time_ms = dataloader.dataset.getitem_time_ms
    training_time_ms = overall_time_ms - dataloading_time_ms

    print('[Process %i] Training Utilization = %.2f%% (%i msec / %i msec)'
          % (args.process_id,
             100. * training_time_ms / overall_time_ms,
             training_time_ms, overall_time_ms))
    if args.featurization == 'dict_proxy' or args.featurization == 'mmap':
        print('[Process %i] Hit %i times; Miss %i times'
              % (args.process_id,
                 dataloader.dataset.num_hit,
                 dataloader.dataset.num_miss))

        total_num_feature = args.max_batches_per_epoch * args.train_batch_size
        print('[Process %i] Feature hit ratio = %.2f%%'
              % (args.process_id,
                 100. * dataloader.dataset.num_hit / total_num_feature))


def train_(args: Namespace,
           model: nn.Module,
           optim: Optimizer,
           dataset: Dataset,
           shared_dict: iter = None):

    if args.featurization == 'mmap' or args.featurization == 'dict_proxy':
        num_features_per_process = \
            int(args.train_batch_size / args.num_processes)
    else:
        num_features_per_process = args.train_batch_size

    np.random.seed(args.process_id)

    # Alternative training function that uses batch-wise parallelism
    start_ms = int(round(time.time() * 1000))
    dataloading_time_ms = 0
    training_time_ms = 0

    model.train()
    for batch_index in range(args.max_batches_per_epoch):

        dataloading_start_ms = int(round(time.time() * 1000))
        # Compute the features and put them into shared_dict
        data_indices = np.random.randint(0, len(dataset),
                                         size=num_features_per_process)

        feature_slice = []
        target_slice = []
        for data_index in data_indices:
            f, t = dataset[data_index]
            feature_slice.append(f)
            target_slice.append(t)

        feature_slice = np.array(feature_slice, dtype=np.float32)
        target_slice = np.array(target_slice, dtype=np.float32)

        # Store features into shared_dict
        # Get the features from shared_dict
        if args.featurization == 'mmap':

            shared_dict[args.process_id]: mmap

            shared_dict[args.process_id].seek(0)
            curr_bytes: bytes = shared_dict[args.process_id].read()
            curr_dict: dict = pickle.loads(curr_bytes)

            # Delete expired features
            for k in list(curr_dict):
                if (batch_index - k) > c.SHARED_DICT_TIMEOUT_BATCH:
                    del curr_dict[k]

            # Add current/new features
            curr_dict[batch_index] = (feature_slice, target_slice)

            new_bytes = pickle.dumps(curr_dict)
            shared_dict[args.process_id].seek(0)
            shared_dict[args.process_id].write(
                new_bytes + b'\0' * (len(new_bytes) - len(curr_bytes)))

            if args.debug:
                print('[Process %i] Completed feature dict'
                      % args.process_id)

            # Now get the features from other shared dict
            feature = np.array([]).reshape(-1, feature_slice.shape[1])
            target = np.array([]).reshape(-1, target_slice.shape[1])
            for i in range(args.num_processes):

                if args.debug:
                    print('[Process %i] Getting features from %i'
                          % (args.process_id, i))

                if i == args.process_id:
                    feature = np.vstack((feature, feature_slice))
                    target = np.vstack((target, target_slice))
                else:
                    shared_dict[i]: mmap
                    while True:
                        try:
                            shared_dict[i].seek(0)
                            tmp_bytes: bytes = shared_dict[i].read()
                            tmp_dict: dict = pickle.loads(tmp_bytes)

                            if batch_index in tmp_dict:
                                f, t = tmp_dict[batch_index]
                                feature = np.vstack((feature, f))
                                target = np.vstack((target, t))
                                break
                            else:
                                time.sleep(0.0001)
                        except:
                            continue

        elif args.featurization == 'dict_proxy':

            shared_dict[args.process_id]: DictProxy

            # Delete expired features
            for k in shared_dict[args.process_id].keys():
                if (batch_index - k) > c.SHARED_DICT_TIMEOUT_BATCH:
                    del shared_dict[args.process_id][k]

            # Add current/new features
            shared_dict[args.process_id][batch_index] = \
                (feature_slice, target_slice)

            # Now get the features from other shared dict
            feature = np.array([]).reshape(-1, feature_slice.shape[1])
            target = np.array([]).reshape(-1, target_slice.shape[1])
            for i in range(args.num_processes):

                if i == args.process_id:
                    feature = np.vstack((feature, feature_slice))
                    target = np.vstack((target, target_slice))
                else:
                    shared_dict[i]: DictProxy
                    while True:

                        if batch_index in shared_dict[i]:
                            f, t = shared_dict[i][batch_index]
                            feature = np.vstack((feature, f))
                            target = np.vstack((target, t))
                            break
                        else:
                            time.sleep(0.0001)
        else:
            feature = feature_slice
            target = target_slice

        dataloading_time_ms += (int(round(time.time() * 1000)) -
                                dataloading_start_ms)

        training_start_ms = int(round(time.time() * 1000))
        feature = torch.from_numpy(feature)
        target = torch.from_numpy(target)

        feature, target = \
            feature.to(args.device, dtype=torch.float32), \
            target.to(args.device, dtype=torch.float32)

        optim.zero_grad()
        prediction = model(feature)
        loss = F.mse_loss(input=prediction, target=target)
        loss.backward()
        optim.step(closure=None)

        training_time_ms += (int(round(time.time() * 1000)) -
                             training_start_ms)

    overall_time_ms = (int(round(time.time() * 1000)) - start_ms)
    # training_time_ms = overall_time_ms - dataloading_time_ms

    print('[Process %i] Training Utilization = %.2f%% (%i msec / %i msec)'
          % (args.process_id,
             100. * training_time_ms / overall_time_ms,
             training_time_ms, overall_time_ms))


def test(args: Namespace,
         model: nn.Module,
         dataloader: DataLoader):
    model.eval()
    with torch.no_grad():
        for feature, target in dataloader:
            feature, target = \
                feature.to(args.device, dtype=torch.float32), \
                target.to(args.device, dtype=torch.float32)

            prediction = model(feature)
            loss = F.mse_loss(input=prediction, target=target)

            # TODO: other metrics here, like R2 scores, MAE, etc.


def start_instance(args: Namespace,
                   shared_dict: DictProxy or mmap or iter = None,
                   shared_lock: Lock = None):
    if args.debug:
        print('[Process %i] MoReL Instance Arguments:\n%s'
              % (args.process_id, json.dumps(vars(args), indent=4)))

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (gpu # or 'cpu')
    use_cuda = (args.device != 'cpu') and torch.cuda.is_available()
    args.device = torch.device(args.device if use_cuda else 'cpu')

    # Data loaders for training/testing #######################################
    dataloader_kwargs = {
        'timeout': 1,
        # 'shuffle': True,
        'pin_memory': True if use_cuda else False,
        'num_workers': c.NUM_DATALOADER_WORKERS if use_cuda else 0}

    train_dataset = ComboDataset(args=args,
                                 training=True,
                                 shared_dict=None,
                                 shared_lock=None)

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     **dataloader_kwargs)

    # test_dataloader = DataLoader(
    #     ComboDataset(args=args,
    #                  training=False,
    #                  shared_dict=shared_dict,
    #                  shared_lock=shared_lock),
    #     batch_size=args.test_batch_size,
    #     **dataloader_kwargs)

    # Constructing neural network and optimizer ###############################
    model = ComboModel(args=args).to(args.device)
    optim = get_optimizer(args=args, model=model)

    # TODO: weight decay and other learning rate manipulation here

    # Training/testing loops ##################################################
    for epoch in range(args.max_num_epochs):

        train_(args=args,
               model=model,
               optim=optim,
               dataset=train_dataset,
               shared_dict=shared_dict)

        # train(args=args,
        #       model=model,
        #       optim=optim,
        #       dataloader=train_dataloader)

        # test(args=args,
        #      model=model,
        #      dataloader=test_dataloader)

    # TODO: summary here? Might have to think about where to put the output
    del model

    if use_cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_args_dict = {
        'process_id': 0,
        'num_processes': 1,
        'rand_state': 0,
        'device': 'cuda:0',

        # Dataloader parameters
        'feature_type': 'ecfp',
        'featurization': 'computing',
        'dict_timeout_ms': c.SHARED_DICT_TIMEOUT_MS,
        'target_dscrptr_name': 'CIC5',

        # Model parameters
        'model_type': 'dense',
        'dense_num_layers': 4,
        'dense_feature_dim': 2048,
        'dense_emb_dim': 4096,
        'dense_dropout': 0.2,

        # Optimizer and other parameters
        'train_batch_size': 32,
        'test_batch_size': 2048,
        'max_num_epochs': 1,
        'max_batches_per_epoch': 100,
        'optimizer': 'sgd',
        'learing_rate': 1e-3,
        'l2_regularization': 1e-5,

        # Debug
        'debug': True,
    }

    test_args = Namespace(**test_args_dict)

    # Configure data loading (featurization) strategy
    # MMAP for shared dict
    # shared = mmap.mmap(fileno=-1,
    #                    length=c.MMAP_BYTE_SIZE,
    #                    prot=mmap.PROT_WRITE)
    # shared.seek(0)
    # shared.write(pickle.dumps({}))
    # test_args.featurization = 'mmap'

    # Dict Proxy for shared dict
    manager = Manager()
    shared = manager.dict()
    test_args.featurization = 'dict_proxy'

    lock = Lock()
    start_instance(args=test_args, shared_dict=[shared, ], shared_lock=lock)
