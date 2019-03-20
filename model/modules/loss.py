# -*- coding: utf-8 -*-
"""
Custom loss

@author: Denis Tome'

"""
import torch

__all__ = [
    'ae_loss'
]


def ae_loss(predicted, target):
    """Custom loss used when both prediction
    and target comes from the model

    Arguments:
        predicted {tensor} -- pytorch tensor
        target {tensor} -- pytorch tensor

    Returns:
        tensor -- loss
    """

    diff = torch.pow(predicted.view_as(target) - target, 2)
    loss = torch.sum(diff, dim=2)
    return torch.mean(loss)
