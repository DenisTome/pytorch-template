# -*- coding: utf-8 -*-
"""
Base dataset class to be extended in dataset_def dir
depending on the different datasets.

@author: Denis Tome'

"""

__version__ = "0.2.0"

from abc import abstractmethod
from enum import Enum, Flag
from torch.utils.data import Dataset
from logger.console_logger import ConsoleLogger
from utils.io import abs_path
import utils.math as umath

__all__ = [
    'SubSet',
    'DatasetInputFormat',
    'BaseDatasetReader',
    'BaseDatasetProxy'
]


class SubSet(Enum):
    """Type of subsets"""

    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'


class DatasetInputFormat(Enum):
    """Data types"""

    LMDB = 'lmdb'
    ORIGINAL = 'orig'
    H5PY = 'h5'
    AWS = 's3'


class OutputData(Flag):
    """Data to return by data loader"""

    IMG = 1 << 0
    P3D = 1 << 1
    P2D = 1 << 2
    ALL = umath.binary_full_n_bits(3)


class BaseDatasetReader(Dataset):
    """Base dataset class"""

    def __init__(self, path: str, sampling: int = 1, desc: str = None):
        """Initialize class

        Args:
            path (str): data path.
            sampling (int, optional): sampling factor. Defaults to None.
            desc (str, optional): description. Defaults to None.
        """

        super().__init__()

        logger_name = self.__class__.__name__
        if desc:
            logger_name += '_{}'.format(desc)

        self._logger = ConsoleLogger(logger_name)
        self._sampling = sampling
        self._path = abs_path(path)

        # ------------------- index data -------------------

        self._indices = self._index_dataset()

    @property
    def sampling(self) -> int:
        """get sampling"""
        return self._sampling

    @property
    def path(self) -> str:
        """get dataset path"""
        return self._path

    @abstractmethod
    def _index_dataset(self) -> list:
        """Index data"""
        raise NotImplementedError

    def __len__(self):
        """Length of the dataset"""
        return len(self._indices)

    def __getitem__(self, index):
        raise NotImplementedError()


class BaseDatasetProxy(Dataset):
    """Base dataset proxy class that loads the dataset base on the input
    source which facilitate having a dataset in different formats based
    on the most ideal configuration for the machine.

    E.g. local v.s. remote execution on AWS.
    """

    def __init__(self,
                 input_type: DatasetInputFormat = DatasetInputFormat.ORIGINAL,
                 out_data_selection: bytes = OutputData.ALL):
        """Initialize class

        Args:
            input_type (DatasetInputFormat, optional): source where to get data from.
                                                       Defaults to DatasetInputFormat.ORIGINAL.
            out_data_selection (bytes, optional): data we want to get as output.
                                                  Defaults to OutputData.ALL.
        """

        super().__init__()

        logger_name = self.__class__.__name__
        self._logger = ConsoleLogger(logger_name)

        self._input_type = input_type
        self._out_data_sel = out_data_selection

    @abstractmethod
    def _get_dataset_reader(self) -> BaseDatasetReader:
        """Based on the value of self._input_type, return the
        correct dataset reader

        Returns:
            BaseDatasetReader: dataset reader to be used
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError()
