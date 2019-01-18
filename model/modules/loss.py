# -*- coding: utf-8 -*-
"""
Created on Jan 18 17:32 2019

@author: Denis Tome'
"""
import torch


def ae_loss(predicted, target):
    """
    Loss to be used specifically when both inputs
    of this functions are output of the network
    (very useful for unsupervised learning)
    :param predicted
    :param target
    :return: loss
    """
    diff = torch.pow(predicted.view_as(target) - target, 2)
    loss = torch.sum(diff, dim=2)
    return torch.mean(loss)
