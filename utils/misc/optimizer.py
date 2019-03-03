""" 
    File Name:          MoReL/optimizer.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               3/3/19
    Python Version:     3.5.4
    File Description:   

"""
from argparse import Namespace
from torch.nn import Module
from torch.optim import Optimizer, Adam, RMSprop, SGD


def get_optimizer(args: Namespace, model: Module) -> Optimizer:

    params = model.parameters()

    if args.optimizer.lower() == 'adam':
        optimizer = Adam(params,
                         amsgrad=True,
                         lr=args.learing_rate,
                         weight_decay=args.l2_regularization)

    elif args.optimizer.lower() == 'rmsprop':
        optimizer = RMSprop(params,
                            lr=args.learing_rate,
                            weight_decay=args.l2_regularization)

    # Use SGD with momentum by default
    else:
        optimizer = SGD(params,
                        momentum=0.8,
                        lr=args.learing_rate,
                        weight_decay=args.l2_regularization)

    return optimizer
