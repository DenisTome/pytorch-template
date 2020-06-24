# -*- coding: utf-8 -*-
"""
Dataset proxy class

@author: Denis Tome'

"""
import torch
from base.base_dataset import BaseDataset, SubSet
from base.base_dataset import OutputData, DatasetInputFormat
from dataset_def.lmdb import LmdbReader
from dataset_def.h5 import H5Reader
from dataset_def.original import OriginalReader
from utils import config

__all__ = [
    'Dataset'
]


class Dataset(BaseDataset):
    """Dataset proxy class"""

    def __init__(self, path: str,
                 input_type: DatasetInputFormat = DatasetInputFormat.ORIGINAL,
                 transf: dict = None, sampling: int = 1, limit: int = -1,
                 out_data: bytes = OutputData.ALL,
                 set_type=SubSet.TRAIN):
        """Init

        Args:
            path (str): dataset path
            input_type (DatasetInputFormat, optional): source data type.
                                                       Defaults to DatasetInputFormat.ORIGINAL.
            transf (dict, optional): data transformation. Defaults to None.
            sampling (int, optional): data sampling factor. Defaults to 1.
            limit (int, optional): max number of samples to consider.
                                   if -1 no limit. Defaults to -1.
            out_data (OutputData, optional): data to return by tge class.
                                             Defaults to OutputData.ALL.
            set_type (SubSet, optional): set type. Defaults to SubSet.TRAIN.
        """

        desc = None
        if set_type != SubSet.TRAIN:
            desc = set_type.value

        super().__init__()

        self.input_type = input_type
        self.out_data = out_data
        self.limit = limit
        self.max_joints = self.get_max_joints()
        self.max_2d_joints = config.model.openpose.n_joints

        # ------------------- data transformations -------------------

        self.transf = transf
        self._check_transormations()

        # ------------------- dataset reader based on type -------------------

        dataset_reader_class = self._get_dataset_reader()
        self.dataset_reader = dataset_reader_class(path, sampling, desc=desc)

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
            BaseDataset: dataset reader
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

    def _apply_transformations(self, data: dict) -> dict:
        """Apply transformation to data

        Args:
            data (dict): keys are the available data OutputData types

        Returns:
            dict: transformed data
        """

        if not self.transf:
            return data

        return self.transf(data, self.out_data)

    def __getitem__(self, index):
        """Get frame

        Arguments:
            index (int): frame number

        Returns:
            torch.Tensor: 3d joint positions
            torch.Tensor: dataset id
        """

        # ------------------- get data -------------------

        # base on what data we want

        frame = self.dataset_reader[index]
        transformed = self._apply_transformations(frame)

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
