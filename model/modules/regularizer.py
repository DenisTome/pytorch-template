# -*- coding: utf-8 -*-
"""
Created on Jul 18 16:05 2018

@author: Denis Tome'

Regularizers to add to the loss

"""
import torch


__all__ = [
    'limb_length',
]


def limb_length(pred_poses, target_poses):
    mse = torch.nn.MSELoss()
    error = mse(pred_poses, target_poses)
    raise NotImplementedError
