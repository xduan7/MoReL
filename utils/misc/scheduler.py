""" 
    File Name:          MoReL/scheduler.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               4/17/19
    Python Version:     3.5.4
    File Description:   

"""
import math

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CyclicCosAnnealingLR(_LRScheduler):

    def __init__(self,
                 optimizer: Optimizer,
                 milestones: list,
                 eta_min: float = 1e-6,
                 last_epoch: int = -1):

        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min
        self.milestones = milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for _ in self.base_lrs]

        base_index = sum(self.last_epoch > ms for ms in self.milestones)
        progress = self.last_epoch if base_index == 0 else \
            self.last_epoch - self.milestones[base_index - 1]
        interval = self.milestones[base_index] if base_index == 0 else \
            self.milestones[base_index] - self.milestones[base_index - 1]
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * progress / interval)) / 2
                for base_lr in self.base_lrs]


if __name__ == '__main__':

    import torch.nn as nn
    import torch.optim as optim
    # import matplotlib.pyplot as plt

    net = nn.Sequential(nn.Linear(2, 2))
    opt = optim.SGD(net.parameters(),
                    lr=1e-3,
                    momentum=0.9,
                    weight_decay=0.0005,
                    nesterov=True)
    skd = CyclicCosAnnealingLR(optimizer=opt,
                               milestones=[(2 ** x) * 4 for x in range(30)])
    lr_log = []
    for i in range(1000):
        opt.step()
        skd.step()
        for param_group in opt.param_groups:
            lr_log.append(param_group['lr'])
    # plt.plot(lr_log)
    # plt.show()
