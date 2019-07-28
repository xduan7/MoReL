""" 
    File Name:          MoReL/graph_drug_response.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               7/28/19
    Python Version:     3.5.4
    File Description:   

"""
import torch
import argparse
import numpy as np
from sklearn.metrics import r2_score
from torch import nn
import torch.nn.functional as F
from typing import List

import sys
sys.path.extend(['/raid/xduan7/Projects/MoReL'])


def main():

    parser = argparse.ArgumentParser(description='Cross Study')

    parser.add_argument('--train_on', type=str, required=True, nargs='+',
                        choices=DATA_SOURCES)
    parser.add_argument('--test_on', type=str, required=True, nargs='+',
                        choices=DATA_SOURCES)

    parser.add_argument('--subsample_on', type=str, required=True,
                        choices=['cell', 'drug'])
    parser.add_argument('--lower_percentage', type=float, required=True)
    parser.add_argument('--higher_percentage', type=float, required=True)
    parser.add_argument('--percentage_increment', type=float, default=0.05)
    parser.add_argument('--state_dim', type=int, default=512)

    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--rand_state', type=int, default=0,
                        help='random state of numpy/sklearn/pytorch')

    args = parser.parse_args()

    device = torch.device(f'cuda: {args.cuda_device}')
    seed_random_state(args.rand_state)

    subsample_percentage_array = np.arange(
        start=args.lower_percentage,
        step=args.percentage_increment,
        stop=args.higher_percentage + .01)

    for subsample_percentage in subsample_percentage_array:
        run_instance(trn_sources=args.train_on,
                     tst_sources=args.test_on,
                     state_dim=args.state_dim,
                     subsample_on=args.subsample_on,
                     subsample_percentage=subsample_percentage,
                     device=device)


if __name__ == '__main__':
    main()