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
import torch
import json
from argparse import Namespace
from mmap import mmap
from multiprocessing.managers import DictProxy

import utils.data_prep.config as c
from networks.combo_model import ComboModel
from utils.datasets.combo_dataset import ComboDataset
from utils.misc.optimizer import get_optimizer
from utils.misc.random_seeding import seed_random_state


def train():
    pass


def test():
    pass


def start(args: Namespace,
          shared_dict: DictProxy or mmap = None):

    print('MoReL Instance Arguments:\n' + json.dumps(vars(args), indent=4))

    # Setting up random seed for reproducible and deterministic results
    seed_random_state(args.rand_state)

    # Computation device config (gpu with # or cpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')

    # Data loaders for training/testing #######################################
    dataloader_kwargs = {
        'timeout': 1,
        'shuffle': 'True',
        'pin_memory': True if use_cuda else False,
        'num_workers': c.NUM_DATALOADER_WORKERS if use_cuda else 0}

    train_dataloader = torch.utils.data.DataLoader(
        ComboDataset(args=args, training=True, shared_dict=shared_dict),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    test_dataloader = torch.utils.data.DataLoader(
        ComboDataset(args=args, training=False, shared_dict=shared_dict),
        batch_size=args.trn_batch_size,
        **dataloader_kwargs)

    # Constructing neural network and optimizer ###############################
    model = ComboModel(args=args).to(device)
    optim = get_optimizer(args=args, model=model)

    # TODO: weight decay and other learning rate manipulation here

    # Training/testing loops ##################################################
    for epoch in range(args.max_num_epochs):

        loss = train()

        loss, result = test()

    # TODO: summary here? Might have to think about where to put the output

