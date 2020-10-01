# -*- coding: utf-8 -*-
"""
Dataset loader class to load data the dataset from the original files.

Here I show an example of the OriginalReader implemented for reading
files from the Human3.6M dataset.

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
import os
import numpy as np
import scipy.io as sio
from base import BaseDatasetReader, OutputData
from utils import io


__all__ = [
    'OriginalReader'
]


class OriginalReader(BaseDatasetReader):
    """Original-files dataset reader"""

    def __init__(self, *args, **kwargs):
        """Init"""

        super().__init__(*args, **kwargs)

        # ------------------- cache -------------------

        self._cache_file_path = None
        self._cache_file_content = None

    def _index_sub_dir(self, path) -> list:
        """Index given directory

        This is specific to each dataset; each has a different
        structure.

        Args:
            path (str): directory path

        Returns:
            list: indexed file paths ([file_path, internal_index])
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
            data = self._index_sub_dir(d)
            indexed.extend(data)

        return indexed

    def _index_dataset(self) -> list:
        """Index dataset files for faster data loading

        Returns:
            list: list of indices ([file_path, internal_index])
        """

        index_file_path = os.path.join(self._path, 'index.npy')
        if io.file_exists(index_file_path):
            self._logger.info(
                'Loading index file {}...'.format(index_file_path))
            indices = np.load(index_file_path, allow_pickle=True)
            return indices

        self._logger.info('Indexing Human3.6M dataset files...')
        indices = self._index_sub_dir(self._path)

        self._logger.info('Saving indexed dataset...')
        np.save(index_file_path, indices, allow_pickle=True)

        return indices

    def _get_processed_sample(self, file_path: str, sample_id: int):
        """Process sample

        Args:
            file_path (str): file path
            sample_id (int): sample id

        Returns:
            np.ndarray: pose
        """

        # Human3.6M stores multiple frames per file
        # if the correct file is already open, avoid loading it again
        if self._cache_file_path is None:
            self._cache_file_path = file_path
            self._cache_file_content = sio.loadmat(
                os.path.join(self.path, file_path))
        else:
            if self._cache_file_path != file_path:
                self._cache_file_path = file_path
                self._cache_file_content = sio.loadmat(
                    os.path.join(self.path, file_path))

        sample = np.array(self._cache_file_content['pose3d_world'][sample_id])

        return sample

    def __getitem__(self, index: int) -> dict:
        """Get sample

        Args:
            index (int): sample id

        Returns:
            dict: dict with frame data
        """

        rel_path, internal_idx = self._indices[index]

        # make absolute path
        path_sample = os.path.join(self.path, str(rel_path))

        frame = self._initialize_frame_output()
        p3d = self._get_processed_sample(path_sample, internal_idx)

        # -------------------------------------------------------- #
        # similarly, gather all the information that the dataset
        # can return, and assigne it according to the name
        # -------------------------------------------------------- #

        frame[OutputData.P3D] = p3d

        return frame
