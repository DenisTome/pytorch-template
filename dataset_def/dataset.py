# -*- coding: utf-8 -*-
"""
Created on Jan 18 17:32 2019

@author: Denis Tome'

Example of Dataset definition class

"""
import utils
from base.base_dataset import BaseDataset

__all__ = [
    'Dataset'
]


class Dataset(BaseDataset):

    def __init__(self, data_path, transform=None):
        """
        :param data_path: Path to data either as file or dir
        :param transform: transformations to apply to the data
        """
        super().__init__(data_path)

        # Load data from file or dir
        self.data_files = None

        self.transform = transform
        self.batch_idx = 0

    def __len__(self):
        return len(self.data_files)

    def _get_data(self, idx):
        """
        Load data from the list of possible samples
        in position idx.
        :param idx: index of the data
        :return: processed data
        """
        data = utils.read_h5(self.data_files[idx])

        # just an example
        img = data['img']
        label = data['label']

        return img, label

    def __getitem__(self, idx):
        img, gt = self._get_data(idx)

        if self.transform:
            transformed = self.transform({'img': img,
                                          'gt': gt})
            img, gt = transformed['img'], transformed['gt']

        return img, gt
