# -*- coding: utf-8 -*-
"""
Multi-class dataset loader to load data of different datasets
from their lmdb files.

Assumption: Lmdb path contains the dataset name in it.
Currently supported ones are:
- Human3.6M (http://vision.imar.ro/human3.6m/description.php)
- CMU Panoptic (http://domedb.perception.cs.cmu.edu/)
- etc...

Usage:
Lmdb format is fast for local machine reading operations but it
is not suitable for cloud service.

@author: Denis Tome'

"""
import lmdb
import torch
from base import BaseDataset
from utils import config, skeletons, io

__all__ = [
    'LmdbReader'
]


class LmdbReader(BaseDataset):
    """Lmdb multi-dataset reader"""

    _H36M = config.dataset.h36m.alias
    _CMU = config.dataset.cmu.alias

    def __init__(self, paths, sampling: int = 1, desc: str = None):
        """Initi multiclass lmdb data loader

        Arguments:
            paths {list} -- list of paths to lmdb datasets

        Keyword Arguments:
            sampling {int} -- sampling factor
        """

        super().__init__(sampling=sampling, desc=desc)

        if not isinstance(paths, list):
            paths = [paths]

        # ------------------- generic -------------------
        self.d_names = self.get_dataset_types(paths)
        self.max_joints = self.get_max_joints()

        # ------------------- lmdb initialization -------------------
        self._logger.info('Initializing datasets from lmdbs...')
        self.envs, self.txns = self._init_lmdb(paths)

        # ------------------- indexing -------------------
        self.num_elems, self.frame_data_map, self.weights = self.index_dataset()

    @staticmethod
    def _init_lmdb(paths):
        """Initialize lmdb datasets for all the provided
        data paths.
        Note: data path is used to generate self.d_names, therefore
        the other of datasets is the same.

        Arguments:
            paths {list} -- paths to lmdb datasets

        Returns:
            lmdb.Environment -- lmdb environment
            lmdb.Transaction -- lmdb transaction to move in the environment
        """

        envs = []
        txns = []
        for did, path in enumerate(paths):
            envs.append(lmdb.open(path, readonly=True))
            txns.append(envs[did].begin())

        return envs, txns

    def index_dataset(self):
        """Index data from the different lmdbs.
        This will allow to choose from whch dataset to sample
        when given a frame id

        Returns:
            int -- total number of elements
            list -- map between frame id and datasets
            list -- weight per dataset
        """

        num_elems = 0
        idx_dataset_map = []
        for did, txn in enumerate(self.txns):
            curr_size = io.unserialize(txn.get('len'.encode('ascii')))
            idx_dataset_map.append({
                'min_id': num_elems,
                'max_id': num_elems + curr_size - 1,
                'did': did
            })
            num_elems += curr_size

        # weithing factor to normalize the difference in dataset sizes
        # which has to be proportionally opposite to the number of elements
        weights = []
        for mapping in idx_dataset_map:
            curr_size = (mapping['max_id'] + 1) - mapping['min_id']
            if curr_size == num_elems:
                curr_weight = 1.0
            else:
                curr_weight = 1 - curr_size / num_elems

            # account for different number of joints
            d_name = self.d_names[mapping['did']]
            n_joints = skeletons[d_name].n_joints
            weight_joints = self.max_joints / n_joints

            curr_weight *= weight_joints
            weights.append(curr_weight)

        return num_elems, idx_dataset_map, weights

    @staticmethod
    def _process_h36m(data, did):
        """Pre-process pose

        Arguments:
            data {dict} -- frame information
            did {int} -- dataset id

        Returns:
            torch.tensor -- 3d joint positions
            torch.tensor -- 3d local joint rotations
            torch.tensor -- 3d root joint translation
            int -- dataset id
        """

        p3d = data.p3d[:skeletons.h36m.n_joints]

        # NOTE: rotations are zeros in h36m
        rot = None

        if data.t is not None:
            root_t = data.t
        else:
            root_t = torch.zeros([1, 3])

        return p3d, rot, root_t, did

    @staticmethod
    def _process_cmu(data, did):
        """Pre-process pose

        Arguments:
            data {dict} -- frame information
            did {int} -- dataset id

        Returns:
            torch.tensor -- 3d joint positions
            torch.tensor -- 3d local joint rotations
            torch.tensor -- 3d root joint translation
            int -- dataset id
        """

        if data.t is not None:
            root_t = data.t
        else:
            root_t = torch.zeros([1, 3])

        return data.p3d[:, :3], data.rot, root_t, did

    def __getitem__(self, index):
        """Get frame

        Arguments:
            index {int} -- frame number

        Returns:
            torch.tensor -- 3d joint positions
            torch.tensor -- 3d local joint rotations
            torch.tensor -- 3d root joint translation
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

        # ------------------- get data from lmdb -------------------
        frame_key = 'frame_{:09d}'.format(fid).encode('ascii')
        data = io.unserialize(self.txns[did].get(frame_key))

        # ------------------- process based on dataset -------------------
        if self.d_names[did] == self._H36M:
            return self._process_h36m(data, did)

        assert self.d_names[did] == self._CMU
        return self._process_cmu(data, did)

    def __len__(self):
        """Get number of elements in the dataset"""

        return self.num_elems
