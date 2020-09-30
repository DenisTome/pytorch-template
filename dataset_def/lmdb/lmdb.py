# -*- coding: utf-8 -*-
"""
MDataset loader class to load data the dataset from the lmdb files.

Here I show an example of the LmdbReader implemented for reading
files from the Human3.6M dataset.

NOTE: the format of the lmdb dataset is using the utility to convert
the original files in lmdb format, therefore we have a defined structure
we have defined when saving the files.

@author: Denis Tome'

"""
import lmdb
from base import BaseDatasetReader, OutputData
from utils import io

__all__ = [
    'LmdbReader'
]


class LmdbReader(BaseDatasetReader):
    """Lmdb dataset reader"""

    def __init__(self, *args, **kwargs):
        """Initi"""

        self._logger.info('Initializing datasets from lmdbs...')
        self._env, self._txn = self._init_lmdb(args[0])

        super().__init__(*args, **kwargs)

    @staticmethod
    def _init_lmdb(path):
        """Init lmdb

        Args:
            path (str): path

        Returns:
            lmdb.Environment: environment
            lmdb.Transaction: pointer
        """

        env = lmdb.open(path, readonly=True)
        txn = env.begin()

        return env, txn

    def _index_dataset(self) -> list:
        """Index data from lmdb.

        Returns:
            list: list of indices
        """

        size = io.unserialize(self._txn.get('len'.encode('ascii')))
        indices = range(size)

        return indices

    def _process(self, data) -> dict:
        """Pre-process pose

        Args:
            data (dict): frame information

        Returns:
            dict: dict with frame data
        """

        p3d = data.p3d

        frame = self._initialize_frame_output()
        frame[OutputData.P3D] = p3d

        # -------------------------------------------------------- #
        # similarly, gather all the information that the dataset
        # can return, and assigne it according to the name
        # -------------------------------------------------------- #

        return frame

    def __getitem__(self, index: int) -> dict:
        """Get sample

        Args:
            index (int): sample id

        Returns:
            dict: dict with frame data
        """

        fid = self._indices[index]

        # ------------------- get data from lmdb -------------------

        frame_key = 'frame_{:09d}'.format(fid).encode('ascii')
        data = io.unserialize(self._txn.get(frame_key))

        return self._process(data)
