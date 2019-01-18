# -*- coding: utf-8 -*-
"""
Created on Jan 18 17:51 2019

@author: Denis Tome'
"""
import numpy as np

__all__ = [
    'compute_3d_joint_error'
]


def compute_3d_joint_error(predicted, gt):
    """
    Compute 3d error per joint in a pose wrt gt
    :param predicted: 3D joint pose linearized [x, y, z ...] or (N, 3) or (3, N)
    :param gt: 3D joint pose linearized [x, y, z ...] or (N, 3) or (3, N)
    :return: joint error
    """
    if type(predicted) is not np.ndarray:
        predicted = np.array(predicted)

    if type(gt) is not np.ndarray:
        gt = np.array(gt)

    if predicted.ndim == 1:
        predicted = np.reshape(predicted, [-1, 3])

    if gt.ndim == 1:
        gt = np.reshape(gt, [-1, 3])

    assert (np.min(predicted.shape) == 3)
    assert (np.min(gt.shape) == 3)

    if predicted.shape[1] != 3:
        predicted = np.transpose(predicted, [1, 0])

    if gt.shape[1] != 3:
        gt = np.transpose(gt, [1, 0])

    diff = predicted - gt
    error = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)

    return error
