# -*- coding: utf-8 -*-
"""
Created on Jan 18 16:47 2019

@author: Denis Tome'

Base dataset class to be extended in dataset_def dir
depending on the different datasets.
"""
import utils
from base.template import FrameworkClass
from torch.utils.data import Dataset


class BaseDataset(FrameworkClass, Dataset):
    """
    Base class for all datasets
    """
    def __init__(self, path):
        super().__init__()
        self.path = utils.abs_path(path)
        self.data_dir = utils.get_dir(path)

    def __iter__(self):
        return NotImplementedError

    def __next__(self):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError
