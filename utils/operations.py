# -*- coding: utf-8 -*-
"""
Specific operations

@author: Denis Tome'

"""
import numpy as np

__all__ = [
    'compute_3d_joint_error'
]


def compute_3d_joint_error(predicted, gt):
    """Compute 3D pose error

    Arguments:
        predicted {numpy array} -- predicted pose
        gt {numpy array} -- ground truth pose

    Returns:
        float -- error
    """

    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    if predicted.ndim == 1:
        predicted = np.reshape(predicted, [-1, 3])

    if gt.ndim == 1:
        gt = np.reshape(gt, [-1, 3])

    assert np.min(predicted.shape) == 3
    assert np.min(gt.shape) == 3

    if predicted.shape[1] != 3:
        predicted = np.transpose(predicted, [1, 0])

    if gt.shape[1] != 3:
        gt = np.transpose(gt, [1, 0])

    diff = predicted - gt
    error = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)

    return error
