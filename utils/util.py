# -*- coding: utf-8 -*-
"""
Created on Jun 08 15:16 2018

@author: Denis Tome'
"""
import re
import os
from copy import copy
import numpy as np

__all__ = [
    'ensure_dir',
    'check_different',
    'is_model_parallel',
    'split_validation'
]


def ensure_dir(path):
    """
    Make sure that directory exists at the specified
    path. If it doesn't, it's created.

    :param path: path of the directory to check
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_different(list_a, list_b):
    """
    Given two lists it identifies the files that are different
    assuming one list is a sub-set of the other
    :param list_a
    :param list_b
    :return: list
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
    """
    Check if a model has been saved in the checkpoint as a DataParallel
    object or a simple model. This changes tha naming convention
    for the layers: module.layer_name instead of layer_name
    :param checkpoint: dictionary with all the model info
    :return: True if saved as DataParallel
    """
    saved_name = list(checkpoint['state_dict'].keys())[0]
    parallel = len(re.findall('module.*', saved_name))

    return bool(parallel)


def split_validation(data_loader, validation_split, randomized=True):
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
