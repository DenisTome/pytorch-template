# -*- coding: utf-8 -*-
"""
Dataset proxy class

@author: Denis Tome'

"""
from enum import Enum, Flag
import torch
from base import BaseDataset
from dataset_def.lmdb import LmdbReader
from dataset_def.h5 import H5Reader
from dataset_def.original import OriginalReader
import utils.math as umath
from utils import config

__all__ = [
    'DatasetInputFormat',
    'OutputData',
    'Dataset',
    'SetType'
]


class OutputData(Flag):
    """Data to return by data loader"""

    P3D = 1 << 0
    DID = 1 << 1
    ROT = 1 << 2
    TRS = 1 << 3
    ALL = umath.binary_full_n_bits(4)


class SetType(Enum):
    """Data to return by data loader"""

    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'


class Dataset(BaseDataset):
    """Dataset proxy class"""

    def __init__(self, paths: list,
                 input_type: DatasetInputFormat = DatasetInputFormat.ORIGINAL,
                 transf: dict = None, sampling: int = 1, limit: int = -1,
                 out_data: bytes = OutputData.ALL,
                 set_type=SetType.TRAIN):
        """Init

        Arguments:
            paths {list} -- dataset paths

        Keyword Arguments:
            input_type {DatasetInputFormat} -- source data type
                                               (default: {DatasetInputFormat.ORIGINAL})
            transf {dict} -- data transformations per dataset, in the format
                            {'d_type': transformation_function}
            sampling {int} -- data sampling (default: {1})
            limit {int} -- number of samples; if -1 no limit (default: {-1})
            out_data {OutputDta} -- data to return by the class
                                    (default: {OutputData.ALL})
            train_set {SetType} -- is training set (Default {SetType.TRAIN})
        """

        desc = None
        if set_type != SetType.TRAIN:
            desc = set_type.value

        super().__init__()

        self.input_type = input_type
        self.out_data = out_data
        self.limit = limit

        if not isinstance(paths, list):
            paths = [paths]

        self.max_joints = self.get_max_joints()

        # ------------------- data transformations -------------------
        self.transf = transf
        self._check_transormations()

        # ------------------- dataset reader based on type -------------------
        dataset_reader_class = self._get_dataset_reader()
        self.dataset_reader = dataset_reader_class(paths, sampling, desc=desc)

    def _check_transormations(self):
        """Check that transormations are in the right format"""

        if self.transf:
            if not isinstance(self.transf, dict):
                self._logger.error('Transformations needs to be a dictionary!')

            if not set(self.transf.keys()).issubset(set(config.dataset.supported)):
                self._logger.error(
                    'Dataset transformations from unsupported dataset!')

    def _get_dataset_reader(self):
        """Get dataset reader based on mode

        Returns:
            BaseDataset -- dataset reader
        """

        if self.input_type == DatasetInputFormat.ORIGINAL:
            return OriginalReader

        if self.input_type == DatasetInputFormat.LMDB:
            return LmdbReader

        if self.input_type == DatasetInputFormat.H5PY:
            return H5Reader

        self._logger.error('Dataset input type not recognized!')
        return -1

    @property
    def d_names(self) -> list:
        """Get daset names"""

        return self.dataset_reader.d_names

    def _apply_transformation_to_generic(self, data, d_name, scope):
        """Run transformation on data

        Arguments:
            data {tensor} -- data
            d_name {str} -- dataset name
            scope {OutputData} -- which type of data is given

        Returns:
            tensor -- processed data
        """

        if not self.transf:
            return data

        if d_name not in list(self.transf.keys()):
            return data

        return self.transf[d_name](data, scope)

    def __getitem__(self, index):
        """Get frame

        Arguments:
            index {int} -- frame number

        Returns:
            torch.tensor -- 3d joint positions
            torch.tensor -- dataset id
        """

        # ------------------- get data -------------------

        # base on what data we want

        p3d, rot, trs, did = self.dataset_reader[index]
        frame = {
            OutputData.P3D: p3d,
            OutputData.ROT: rot
        }

        # ------------------------------------------------------
        # ------------------- data selection -------------------
        # ------------------------------------------------------

        out = []

        # ------------------- p3d -------------------

        if bool(self.out_data & OutputData.P3D):

            trsf_p3d = self._apply_transformation_to_generic(
                frame, self.d_names[did], OutputData.P3D)

            if trsf_p3d.shape[0] != self.max_joints:
                p3d_padding = torch.zeros([self.max_joints, 3])
                p3d_padding[:trsf_p3d.shape[0]] = trsf_p3d
                out.append(p3d_padding.float())
            else:
                out.append(trsf_p3d.float())

        # ------------------- dataset id -------------------

        if bool(self.out_data & OutputData.DID):
            out.append(did)

        # ------------------- rotation -------------------

        if bool(self.out_data & OutputData.ROT):

            trsf_rot = self._apply_transformation_to_generic(
                frame, self.d_names[did], OutputData.ROT)

            if trsf_rot.shape[0] != self.max_joints:

                if len(list(trsf_rot.shape)) == 2:
                    # quaternion representation
                    rot_padding = torch.zeros([self.max_joints, 4])
                else:
                    # matrix representation
                    rot_padding = torch.zeros([self.max_joints, 3, 3])

                rot_padding[:trsf_rot.shape[0]] = trsf_rot
                out.append(rot_padding.float())
            else:
                out.append(trsf_rot.float())

        # ------------------- root translation -------------------

        if bool(self.out_data & OutputData.TRS):

            out.append(trs.float())

        if len(out) == 1:
            return out[0]

        return out

    def __len__(self):
        if self.limit > 0:
            return self.limit

        return len(self.dataset_reader)
