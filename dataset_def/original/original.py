# -*- coding: utf-8 -*-
"""
Multi-class dataset loader to load data of different datasets
from their original files.

Currently supported ones are:
- Human3.6M (http://vision.imar.ro/human3.6m/description.php)
- CMU Panoptic (http://domedb.perception.cs.cmu.edu/)
- ...

Each custom dataset parser returns a list of 2 elements:
1. relative path (wrt to dataset root dir) of the file
2. frame index
where the frame index is important since in many datasets, multiple
frames are stored in the same file.

For performance issues related to the different configuraionts, we
might want to split them or keep them together (i.e. local vs. cloud
computing).

Relative paths are important in case we want to change the location
of the dataset directory without compromizing the index.h5 file.
This file contains the relative path of all the elements contrained in
the dataset such that the loading time is drastically reduced.

@author: Denis Tome'

"""
import os
from base import BaseDataset
from dataset_def.original import H36mParser, CMUParser
from base.base_dataset import OutputData, DatasetInputFormat
from utils import config

__all__ = [
    'OriginalReader'
]


class OriginalReader(BaseDataset):
    """Original file multi-dataset reader"""

    def __init__(self, paths, sampling: int = 1):
        """Initi multiclass original data loader

        Arguments:
            paths {list} -- list of paths to datasets dirs

        Keyword Arguments:
            sampling {int} -- sampling factor
        """

        super().__init__(sampling=sampling)

        if not isinstance(paths, list):
            paths = [paths]
        self.paths = paths

        # ------------------- generic -------------------
        self.d_names = self.get_dataset_types(paths)
        self.max_joints = self.get_max_joints()

        # ------------------- indexing -------------------
        self._logger.info('Initializing datasets from oringal files...')
        self.dataset_parsers, self.frame_data_map, self.num_elems = self.index_datasets()

    def index_datasets(self):
        """Index datasets

        Add here all dataset parsers, who's job is to go through the
        original dataset format, file by file, and get a list of 
        indices, frame_id to be able to parse the data in the data loader

        Returns:
            dict -- dataset parsers
            list -- map between frame id and datasets
        """

        dataset_parsers = []
        num_elems = 0
        idx_dataset_map = []

        # ------------------- index datasets separately -------------------
        for did, (d_name, path) in enumerate(zip(self.d_names, self.paths)):

            if d_name == config.dataset.h36m.alias:
                dataset_parsers.append(H36mParser(path,
                                                  sampling=self.sampling,
                                                  root_norm=False))
            elif d_name == config.dataset.cmu.alias:
                dataset_parsers.append(CMUParser(path,
                                                 sampling=self.sampling,
                                                 root_norm=True))
            else:
                self._logger.error('Dataset not supported!')

            index = dataset_parsers[did].index()
            idx_dataset_map.append({
                'min_id': num_elems,
                'max_id': num_elems + len(index) - 1,
                'did': did
            })
            num_elems += len(index)

        return dataset_parsers, idx_dataset_map, num_elems

    def __getitem__(self, index):
        """Get sample

        Arguments:
            index (int): sample id

        Returns:
            dict: dict with frame data
        """

        # get corresponding dataset id for given index
        did = 0
        for fm in self.frame_data_map:
            if index in range(fm['min_id'], fm['max_id'] + 1):
                break
            did += 1

        # ------------------- dataset specific information -------------------
        fid = index - self.frame_data_map[did]['min_id']
        # multiple samples can be stored in the same file
        rel_path, internal_idx = self.dataset_parsers[did].get_element(fid)

        # make absolute path
        sample_path = os.path.join(self.paths[did], rel_path)

        # points, root rotation and translation
        p3d, rot, t = self.dataset_parsers[did].process(sample_path,
                                                        internal_idx)

        frame = self.initialize_frame_output()
        frame[OutputData.P3D] = p3d
        frame[OutputData.ROT] = rot
        frame[OutputData.T] = t

        return frame

    def __len__(self):
        """Get number of elements in the dataset"""

        return self.num_elems
