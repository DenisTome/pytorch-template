# -*- coding: utf-8 -*-
"""
Created on Jun 08 15:00 2018

@author: Denis Tome'
"""

__all__ = [
    'get_optimizer_lr'
]


def get_optimizer_lr(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr