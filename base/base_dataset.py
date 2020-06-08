# -*- coding: utf-8 -*-
"""
Base dataset class to be extended in dataset_def dir
depending on the different datasets.

@author: Denis Tome'

"""
from enum import Enum, auto
from torch.utils.data import Dataset
import numpy as np
from logger.console_logger import ConsoleLogger
from utils.config import config, skeletons
from utils.io import abs_path

__all__ = [
    'BaseDataset',
    'SubSet',
    'DatasetInputFormat'
]


class SubSet(Enum):
    """Type of subsets"""
    train = auto()
    test = auto()
    val = auto()


class DatasetInputFormat(Enum):
    """Data types"""
    LMDB = 'lmdb'
    ORIGINAL = 'orig'
    H5PY = 'h5'
    AWS = 's3'


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, path=None, sampling=None, desc=None):

        super().__init__()

        logger_name = self.__class__.__name__
        if desc:
            logger_name += '_{}'.format(desc)

        self._logger = ConsoleLogger(logger_name)
        if path:
            self.data_dir = abs_path(path)
        if sampling:
            self.sampling = sampling
        else:
            self.sampling = 1

    def get_dataset_types(self, paths: list) -> list:
        """Dataset types from lmdb paths

        Arguments:
            paths {list} -- list of lmdb paths

        Returns:
            list -- dataset names
        """

        d_types = []
        for p in paths:
            for d_type in config.dataset.supported:
                if d_type in p:
                    d_types.append(d_type)
                    break

        if len(d_types) != len(paths):
            self._logger.error('Some of lmdb datasets not recognized!')

        return d_types

    @staticmethod
    def get_max_joints() -> int:
        """Get max number of joints for all supported datasets

        Returns:
            int -- maximum number of joints
        """

        n_joints = []
        for d_name in config.dataset.supported:
            n_joints.append(skeletons[d_name].n_joints)

        return np.array(n_joints).max()

    @staticmethod
    def get_max_limbs() -> int:
        """Get max number of joints for all supported datasets

        Returns:
            int -- maximum number of joints
        """

        n_limbs = []
        for d_name in config.dataset.supported:
            n_limbs.append(skeletons[d_name].n_limbs)

        return np.array(n_limbs).max()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
