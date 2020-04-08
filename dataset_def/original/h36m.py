# -*- coding: utf-8 -*-
"""
Human3.6m dataset orignal data processing.

The same has to be replicated for each dataset we have and we
want to load.

@author: Denis Tome'

"""
import os
import scipy.io as sio
import numpy as np
from base import BaseDatasetParser
from utils import io

__all__ = [
    'H36mParser'
]


class H36mParser(BaseDatasetParser):
    """H36M Paraser"""

    def __init__(self, *args, **kwargs):
        """Init"""

        super().__init__(*args, **kwargs)
        self.n_joints = None

    def _index(self, path: str) -> list:
        """Index directory

        This is specific to each dataset; each has a different
        structure.

        Arguments:
            path {str} -- directory path

        Returns:
            list -- indexed file paths
        """

        # mocap data is in a leaf folder
        _, dirs = io.get_sub_dirs(path)
        if not dirs:

            # relative path
            r_path = io.make_relative(path, self.path)

            # h36m has one mat file for all images
            data = '{}/h36m_meta.mat'.format(r_path).encode('utf8')
            _, files = io.get_files(path, 'jpg')

            list_data = []
            for i in range(0, len(files), self.sampling):
                list_data.append([data, i])
            return list_data

        # collect from sub-dirs
        indexed = []

        for d in dirs:
            data = self._index(d)
            indexed.extend(data)

        return indexed

    def index(self) -> list:
        """Index dataset and load if it already exists

        Returns:
            list -- indices [path, internal_idx]
        """

        index_file_path = os.path.join(self.path, 'index.h5')
        if io.file_exists(index_file_path):
            self._logger.info(
                'Loading Human3.6M index {}...'.format(index_file_path))
            self.indices = io.read_h5(index_file_path)['val']
            return self.indices

        self._logger.info('Indexing Human3.6M dataset files...')
        self.indices = self._index(self.path)

        self._logger.info('Saving indexed dataset...')
        io.write_h5(index_file_path, self.indices)

        return self.indices

    def root_rot_translation(self, data, fid: int):
        """Get root joint rotation and translation

        Arguments:
            data {Bvh} -- data
            fid {int} -- frame id

        Returns:
            np.ndarray -- quaternions expressing joint rotations
            np.ndarray -- root translation
        """

        # Retrieve data here...

        return np.zeros([self.n_joints, 4]), np.zeros(3)

    def process(self, file_path: str, s_id: int):
        """Process sample

        Arguments:
            file_path {str} -- file path
            s_id {int} -- sample id

        Returns:
            np.ndarray -- pose
        """

        # to avoid to load the same file multiple times
        # if the dataset has few big files with many
        # frames
        if not self.cache_file:
            self.cache_file = file_path
            self.cache = sio.loadmat(
                os.path.join(self.path, self.cache_file))
        else:
            if self.cache_file != file_path:
                self.cache_file = file_path
                self.cache = sio.loadmat(
                    os.path.join(self.path, self.cache_file))

        if self.n_joints is None:
            self.n_joints = self.cache['pose3d_world'][0].shape[0]

        r_rot, r_trans = self.root_rot_translation(self.cache, s_id)

        return np.array(self.cache['pose3d_world'][s_id]), r_rot, r_trans
