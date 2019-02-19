# -*- coding: utf-8 -*-
"""
Base dataset class to be extended in dataset_def dir
depending on the different datasets.

@author: Denis Tome'

"""
from torch.utils.data import Dataset
from base.template import FrameworkClass
import utils


class BaseDataset(FrameworkClass, Dataset):
    """Base dataset class"""

    def __init__(self, path):
        super().__init__()
        self.path = utils.abs_path(path)
        self.data_dir = utils.get_dir(path)

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
