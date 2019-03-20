# -*- coding: utf-8 -*-
"""
Base dataset class to be extended in dataset_def dir
depending on the different datasets.

@author: Denis Tome'

"""
import enum
from torch.utils.data import Dataset
from logger.console_logger import ConsoleLogger
import utils

__all__ = [
    'BaseDataset',
    'SubSet'
]


class SubSet(enum.Enum):
    """Type of subsets"""

    train = 0
    test = 1
    val = 2


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, path):
        super().__init__()

        logger_name = self.__class__.__name__
        self._logger = ConsoleLogger(logger_name)

        self.path = utils.abs_path(path)
        self.data_dir = utils.get_dir(path)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
