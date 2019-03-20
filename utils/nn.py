# -*- coding: utf-8 -*-
"""
Custom controllers for training the model

@author: Denis Tome'

"""

__all__ = [
    'get_optimizer_lr'
]


def get_optimizer_lr(optimizer):
    """Get learning-rate from optimizer

    Arguments:
        optimizer {Optimizer} -- torch optimizer

    Returns:
        float -- learning rate
    """

    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr
