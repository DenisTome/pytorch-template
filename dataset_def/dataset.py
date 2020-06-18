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
from base.base_dataset import OutputData, DatasetInputFormat
import utils.math as umath
from utils import config

__all__ = [
    'Dataset'
]


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

    def _apply_transformations(self, data: dict, d_name: str):
        """Apply transformation to data

        Args:
            data (dict): keys are the available data OutputData types
            d_name (str): dataset name

        Returns:
            dict: transformed data
        """

        if not self.transf:
            return data

        if d_name not in list(self.transf.keys()):
            return data

        res = self.transf[d_name](data, self.out_data)
        return res

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

        frame = self.dataset_reader[index]
        transformed = self._apply_transformations(
            frame,
            self.d_names[did])

        # ------------------------------------------------------
        # ------------------- data selection -------------------
        # ------------------------------------------------------

        out = []

        # ------------------- img -------------------

        if bool(self.out_data & OutputData.IMG):
            if transformed[OutputData.IMG] is None:
                self._logger.error('Image not available in data loader')
            out.append(transformed[OutputData.IMG])


         # ------------------- p3d -------------------

        if bool(self.out_data & OutputData.P3D):

            if transformed[OutputData.P3D] is None:
                self._logger.error('P3d not available in data loader')
            trsf_p3d = transformed[OutputData.P3D]

            if trsf_p3d.shape[0] != self.max_joints:
                p3d_padding = torch.zeros([self.max_joints, 3])
                p3d_padding[:trsf_p3d.shape[0]] = trsf_p3d
                out.append(p3d_padding.float())
            else:
                out.append(trsf_p3d.float())

        # ------------------- p2d -------------------

        if bool(self.out_data & OutputData.P2D):

            if transformed[OutputData.P2D] is None:
                self._logger.error('P2d not available in data loader')
            trsf_p2d = transformed[OutputData.P2D]

            if trsf_p2d.shape[0] != self.max_joints:
                p2d_padding = torch.zeros([self.max_joints, 2])
                p2d_padding[:trsf_p2d.shape[0]] = trsf_p2d
                out.append(p2d_padding.float())
            else:
                out.append(trsf_p2d.float())

        # ------------------- rotation -------------------

        if bool(self.out_data & OutputData.ROT):

            if transformed[OutputData.ROT] is None:
                self._logger.error(
                    'ROT hat not available in data loader')
            trsf_rot = transformed[OutputData.ROT]

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

            if transformed[OutputData.TRS] is None:
                self._logger.error(
                    'Root translation hat not available in data loader')
            out.append(trs.float())

        # ------------------- dataset id -------------------

        if bool(self.out_data & OutputData.DID):
            if transformed[OutputData.DID] is None:
                self._logger.error(
                    'DatasetId hat not available in data loader')
            out.append(frame[OutputData.DID])

        # ------------------- meta-data -------------------

        if bool(self.out_data & OutputData.META):
            if transformed[OutputData.META] is None:
                self._logger.error('Metadata hat not available in data loader')
            out.append(frame[OutputData.META])

        if len(out) == 1:
            return out[0]

        return out

    def __len__(self):
        if self.limit > 0:
            return self.limit

        return len(self.dataset_reader)
