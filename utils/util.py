# -*- coding: utf-8 -*-
"""
Utils

@author: Denis Tome'

"""
import re
from itertools import permutations
import numpy as np
from utils import config, skeletons

__all__ = [
    'check_different',
    'is_model_parallel',
    'substract_ranges',
    'get_model_modes',
    'is_model_cycle_mode'
]


def check_different(list_a: list, list_b: list) -> list:
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


def substract_ranges(range_a: range, range_b: range, assume_unique=True) -> list:
    """Subtract ranges

    Arguments:
        range_a {range} -- range a
        range_b {range} -- range b

    Keyword Arguments:
        assume_unique {book} -- assumption (default: {True})

    Returns:
        list -- elements of a not in b
    """

    return np.setdiff1d(range_a, range_b, assume_unique).tolist()


def get_model_modes() -> list:
    """Compute model modes for pose prediction (input pose and output pose)

    Returns:
        list -- possible modes
    """

    # from whatever dataset to same dataset
    modes = ['dataset_to_dataset']

    # from one dataset to another one
    pairs = list(permutations(config.dataset.supported))
    for p in pairs:
        modes.append('{}_to_{}'.format(*p))

    for d_name in config.dataset.supported:

        modes.append('{}_to_z'.format(d_name))
        modes.append('z_to_{}'.format(d_name))

        if skeletons[d_name].rot:
            modes.append('z_to_{}-rot'.format(d_name))

    # from specific dataset to same dataset
    for d_name in config.dataset.supported:

        modes.append('{0}_to_{0}'.format(d_name))

        if skeletons[d_name].rot:
            modes.append('{0}_to_{0}-rot'.format(d_name))

    return modes


def is_model_parallel(checkpoint: dict) -> bool:
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


def is_model_cycle_mode(model_mode: str) -> bool:
    """Check if model is in cycle mode

    Arguments:
        model_mode {str} -- model mode

    Returns:
        bool -- True if in cycle mode
    """

    matches = re.findall(r'_to_', model_mode)
    if not matches:
        return False

    if len(matches) == 2:
        return True

    return False
