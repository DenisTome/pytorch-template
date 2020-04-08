# -*- coding: utf-8 -*-
"""
Multi-class dataset loader to load data from different datasets
from their h5 files.

Currently supported ones are:
- Human3.6M (http://vision.imar.ro/human3.6m/description.php)
- CMU Panoptic (http://domedb.perception.cs.cmu.edu/)

@author: Denis Tome'

"""
import os
import torch
from base import BaseDataset
from utils import config, io

__all__ = [
    'H5Reader'
]


class H5Reader(BaseDataset):
    """H5 multi-dataset reader"""

    def __init__(self, paths, sampling: int = 1, desc: str = None):
        """Initi multiclass hy data loader

        Arguments:
            paths {list} -- list of paths to lmdb datasets

        Keyword Arguments:
            sampling {int} -- sampling factor
        """

        super().__init__(sampling=sampling, desc=desc)

        if not isinstance(paths, list):
            paths = [paths]
        self.paths = paths

        # ------------------- generic -------------------
        self.d_names = self.get_dataset_types(paths)
        self.max_joints = self.get_max_joints()

        # ------------------- indexing -------------------
        self.num_elems, self.frame_data_map = self.index_dataset()

    def index_dataset(self):
        """Index data from the different h5 format datasets.
        This will allow to choose from whch dataset to sample
        when given a frame id

        Returns:
            int -- total number of elements
            list -- map between frame id and datasets
        """

        num_elems = 0
        idx_dataset_map = []
        for did, path in enumerate(self.paths):

            metadata = io.read_from_json(os.path.join(path, 'meta.json'))
            curr_size = int(metadata['len'])

            idx_dataset_map.append({
                'min_id': num_elems,
                'max_id': num_elems + curr_size - 1,
                'did': did
            })
            num_elems += curr_size

        return num_elems, idx_dataset_map

    def __getitem__(self, index):
        """Get frame

        Arguments:
            index {int} -- frame number

        Returns:
            torch.tensor -- 3d joint positions
            torch.tensor -- root joint rotation
            torch.tensor -- root joint translation
            int -- dataset id
        """

        # choose dataset based on the mapping index
        did = 0
        for fm in self.frame_data_map:
            if index in range(fm['min_id'], fm['max_id'] + 1):
                break
            did += 1

        # ------------------- dataset specific information -------------------
        fid = index - self.frame_data_map[did]['min_id']

        # ------------------- get data from h5 file -------------------

        frame_name = 'frame_{:09d}.h5'.format(fid)
        frame_path = os.path.join(self.paths[did], frame_name)
        data = io.read_h5(frame_path)

        # ------------------- get data -------------------

        p3d = torch.tensor(data['p3d'])
        rot = torch.tensor(data['rot'])
        trs = torch.tensor(data['trs'])

        return p3d, rot, trs, did

    def __len__(self):
        """Get number of elements in the dataset"""

        return self.num_elems
