# -*- coding: utf-8 -*-
"""
Utils

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"

import re
import numpy as np
from utils import config

__all__ = [
    'check_different',
    'is_model_parallel',
    'substract_ranges',
    'compute_3d_joint_error'
]


def check_different(list_a: list, list_b: list) -> list:
    """Check differences in two lists

    Args:
        list_a (list): first list
        list_b (list): second list

    Returns:
        list: list containing the differences
    """

    diff = []
    num = np.max([len(list_a), len(list_b)])
    idx_a = 0
    idx_b = 0
    while idx_a < num and idx_b < num:
        if list_a[idx_a] == list_b[idx_b]:
            idx_a += 1
            idx_b += 1
            continue

        if len(list_a) < len(list_b):
            diff.append(list_b[idx_b])
            idx_b += 1
            continue

        diff.append(list_a[idx_a])
        idx_a += 1

    return diff


def substract_ranges(range_a: range, range_b: range, assume_unique: bool = True) -> list:
    """Subtract ranges

    Args:
        range_a (range): range a
        range_b (range): range b
        assume_unique (bool, optional): unique assumption. Defaults to True.

    Returns:
        list: elements of a not in b
    """

    return np.setdiff1d(range_a, range_b, assume_unique).tolist()


def is_model_parallel(checkpoint: dict) -> bool:
    """Check if a model has been saved as parallel

    Args:
        checkpoint (dict): dictionary saved according to the base_trainer format

    Returns:
        bool: True if it is saved as parallel
    """

    saved_name = list(checkpoint['state_dict'].keys())[0]
    parallel = len(re.findall('module.*', saved_name))

    return bool(parallel)


def compute_3d_joint_error(predicted: np.array, gt: np.array) -> float:
    """Compute 3D pose error

    Args:
        predicted (np.array): predicted pose
        gt (np.array): ground truth pose

    Returns:
        float: error
    """

    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    if predicted.ndim == 1:
        predicted = np.reshape(predicted, [-1, 3])

    if gt.ndim == 1:
        gt = np.reshape(gt, [-1, 3])

    # it includes the confidence as well
    assert np.min(predicted.shape) == config.dataset.n_components
    assert np.min(gt.shape) == config.dataset.n_components

    if predicted.shape[1] != config.dataset.n_components:
        predicted = np.transpose(predicted, [1, 0])

    if gt.shape[1] != config.dataset.n_components:
        gt = np.transpose(gt, [1, 0])

    diff = predicted[:, :3] - gt[:, :3]
    error = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)

    return error
