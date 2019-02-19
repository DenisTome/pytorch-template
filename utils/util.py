# -*- coding: utf-8 -*-
"""
Created on Jun 08 15:16 2018

@author: Denis Tome'

"""
import re
from copy import copy
import numpy as np

__all__ = [
    'check_different',
    'is_model_parallel',
    'split_validation'
]


def check_different(list_a, list_b):
    """Check differences in two lists

    Arguments:
        list_a {list} -- first list
        list_b {list} -- second list

    Returns:
        list -- list containing the differences
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


def is_model_parallel(checkpoint):
    """Check if a model has been saved as parallel

    Arguments:
        checkpoint {dict} -- dictionary saved according to
                             the base_trainer format

    Returns:
        bool -- True if it is saved as parallel
    """

    saved_name = list(checkpoint['state_dict'].keys())[0]
    parallel = len(re.findall('module.*', saved_name))

    return bool(parallel)


def split_validation(data_loader, validation_split, randomized=True):
    """Split dataset into train and validation set

    Arguments:
        data_loader {DataLoader} -- pytorch data loader
        validation_split {float} -- percentage of validation set

    Keyword Arguments:
        randomized {bool} -- randomized splitting (default: {True})

    Returns:
        TrainLoader, ValidationLoader -- splitted data loaders
    """

    if validation_split == 0.0:
        return data_loader, None
    valid_data_loader = copy(data_loader)
    if randomized:
        rand_idx = np.random.permutation(len(data_loader.x))
        data_loader.x = np.array([data_loader.x[i] for i in rand_idx])
        data_loader.y = np.array([data_loader.y[i] for i in rand_idx])
    split = int(len(data_loader.x) * validation_split)
    valid_data_loader.x = data_loader.x[:split]
    valid_data_loader.y = data_loader.y[:split]
    data_loader.x = data_loader.x[split:]
    data_loader.y = data_loader.y[split:]
    return data_loader, valid_data_loader
