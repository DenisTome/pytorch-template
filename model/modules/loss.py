# -*- coding: utf-8 -*-
"""
Custom losses

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
import torch

__all__ = [
    'ae_loss'
]


def ae_loss(predicted, target) -> float:
    """Custom loss used when both prediction
    and target comes from the model

    Args:
        predicted (torch.Tensor): pytorch tensor
        target (torch.Tensor): pytorch tensor

    Returns:
        float: loss
    """

    diff = torch.pow(predicted.view_as(target) - target, 2)
    loss = torch.sum(diff, dim=2)
    return torch.mean(loss)
