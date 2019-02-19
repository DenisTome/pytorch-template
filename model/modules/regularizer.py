# -*- coding: utf-8 -*-
"""
Regularizers to add to the loss

@author: Denis Tome'

"""
import torch


__all__ = [
    'limb_length',
]


def limb_length(pred_poses, target_poses):
    """Limb length regularizer"""

    mse = torch.nn.MSELoss()
    error = mse(pred_poses, target_poses)
    raise NotImplementedError
