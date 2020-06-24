# -*- coding: utf-8 -*-
"""
Dataset loader class to load data of different datasets
from their original files.

Currently supported one:
- Human3.6M (http://vision.imar.ro/human3.6m/description.php)

Each custom dataset parser returns a list of 2 elements:
1. relative path (wrt to dataset root dir) of the file
2. frame index
where the frame index is important since in many datasets, multiple
frames are stored in the same file.

Relative paths are important in case we want to change the location
of the dataset directory without compromizing the index.h5 file.
This file contains the relative path of all the elements contrained in
the dataset such that the loading time is drastically reduced.

@author: Denis Tome'

"""
import os
from base.base_dataset import BaseDataset, OutputData
from dataset_def.original import H36mParser

__all__ = [
    'OriginalReader'
]


class OriginalReader(BaseDataset):
    """Original file multi-dataset reader"""

    def __init__(self, path: str, sampling: int = 1):
        """Init

        Args:
            path (str): dataset path
            sampling (int, optional): sampling factor. Defaults to 1.
        """

        super().__init__(sampling=sampling)

        # ------------------- generic -------------------
        self.path = path
        self.max_joints = self.get_max_joints()

        # ------------------- indexing -------------------
        self._logger.info('Initializing datasets from oringal files...')
        self.dataset_parser, self.num_elems = self.index_datasets()

    def index_datasets(self):
        """Index dataset

        This is the function indexing the dataset / multiple datasets.
        The code right now is set only for one dataset, but it can be
        quicly extended for a multi-dataset configuration.

        Returns:
            BaseDatasetParser: dataset parser
            int: number of elements
        """

        dataset_parser = H36mParser(self.path,
                                    sampling=self.sampling,
                                    root_norm=False)

        index = dataset_parser.index()
        num_elems = len(index)

        return dataset_parser, num_elems

    def __getitem__(self, index: int) -> dict:
        """Get sample

        Arguments:
            index (int): sample id

        Returns:
            dict: dict with frame data
        """

        # get corresponding dataset id for given index
        rel_path, internal_idx = self.dataset_parser.get_element(index)

        # make absolute path
        sample_path = os.path.join(self.path, rel_path)

        # points, root rotation and translation
        p3d, rot, _ = self.dataset_parser.process(sample_path,
                                                  internal_idx)

        # initialize dictionary with all returnable data
        # and only provide the information retrievable by
        # this class
        frame = self.initialize_frame_output()
        frame[OutputData.P3D] = p3d
        frame[OutputData.ROT] = rot

        return frame

    def __len__(self):
        """Get number of elements in the dataset"""

        return self.num_elems
