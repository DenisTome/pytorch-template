# -*- coding: utf-8 -*-
"""
Dataset proxy class

@author: Denis Tome'

"""
import base
from base.base_dataset import OutputData, DatasetInputFormat
from dataset_def.lmdb import LmdbReader
from dataset_def.h5 import H5Reader
from dataset_def.original import OriginalReader

__all__ = [
    'Dataset'
]


class Dataset(base.BaseDatasetProxy):
    """Dataset proxy class"""

    def __init__(self, path: str, sampling: int = 1,
                 set_type=base.SubSet.TRAIN, **kwargs):
        """Init

        Args:
            path (str): dataset path
            transf (dict, optional): data transformation. Defaults to None.
            sampling (int, optional): data sampling factor. Defaults to 1.
            set_type (SubSet, optional): set type. Defaults to SubSet.TRAIN.
        """

        desc = None
        if set_type != base.SubSet.TRAIN:
            desc = set_type.value

        super().__init__(**kwargs)

        # ------------------- dataset reader based on type -------------------

        dataset_reader = self._get_dataset_reader()
        self._dataset_reader = dataset_reader(path, sampling,
                                              out_data_selection=self._out_data_sel,
                                              desc=desc)

    def _get_dataset_reader(self) -> base.base_dataset.BaseDatasetReader:
        """Get dataset reader based on mode

        Returns:
            BaseDataset: dataset reader
        """

        if self._input_type == DatasetInputFormat.ORIGINAL:
            return OriginalReader

        if self._input_type == DatasetInputFormat.LMDB:
            return LmdbReader

        if self._input_type == DatasetInputFormat.H5PY:
            return H5Reader

        self._logger.error('Dataset input type not recognized!')
        return -1

    def __getitem__(self, index: int) -> list:
        """Get frame

        Arguments:
            index (int): frame number

        Returns:
            list: output list of processed elements
        """

        # ------------------- get data -------------------

        # base on what data we want

        frame = self._dataset_reader[index]
        transformed = self._apply_transformations(frame)

        # ------------------------------------------------------
        # ------------------- data selection -------------------
        # ------------------------------------------------------

        out = []

        # ------------------- img -------------------

        if bool(self._out_data_sel & OutputData.IMG):
            if transformed[OutputData.IMG] is None:
                self._logger.error('Image not available in data loader')
            out.append(transformed[OutputData.IMG])

         # ------------------- p3d -------------------

        if bool(self._out_data_sel & OutputData.P3D):

            if transformed[OutputData.P3D] is None:
                self._logger.error('P3d not available in data loader')
            trsf_p3d = transformed[OutputData.P3D]
            out.append(trsf_p3d.float())

        # ------------------- p2d -------------------

        if bool(self._out_data_sel & OutputData.P2D):

            if transformed[OutputData.P2D] is None:
                self._logger.error('P2d not available in data loader')
            trsf_p2d = transformed[OutputData.P2D]
            out.append(trsf_p2d.float())

        if len(out) == 1:
            return out[0]

        return out

    def __len__(self):
        return len(self.dataset_reader)
