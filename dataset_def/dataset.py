# -*- coding: utf-8 -*-
"""
Example of Dataset definition class

@author: Denis Tome'

"""
import utils
from base.base_dataset import BaseDataset

__all__ = [
    'Dataset'
]


class Dataset(BaseDataset):
    """Dataset"""

    def __init__(self, path, transform=None):
        """Initialization

        Arguments:
            path {str} -- path to the data

        Keyword Arguments:
            transform {FrameworkClass} -- transformation to apply to
                                          the data (default: {None})
        """

        super().__init__(path)

        # Load data from file or dir
        self.data_files = None

        self.transform = transform
        self.batch_idx = 0

    def __len__(self):
        """Get number of elements in the dataset"""

        return len(self.data_files)

    def _get_data(self, idx):
        """Load data from the list

        Arguments:
            idx {int} -- sample index

        Returns:
            undefined -- whatever is needed
        """

        data = utils.read_h5(self.data_files[idx])

        # just an example
        img = data['img']
        label = data['label']

        return img, label

    def __getitem__(self, idx):
        """Get sample

        Arguments:
            idx {int} -- sample id

        Returns:
            undefined -- whatever is needed
        """

        img, gt = self._get_data(idx)

        if self.transform:
            transformed = self.transform({'img': img,
                                          'gt': gt})
            img, gt = transformed['img'], transformed['gt']

        return img, gt
