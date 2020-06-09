# -*- coding: utf-8 -*-
"""
Base dataset parser to take original dataset specific inputs
and convert them in a standard format usable by the
base dataset classes

@author: Denis Tome'

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"


from abc import abstractmethod
from base.template import FrameworkClass


class BaseDatasetParser(FrameworkClass):
    """BaseDatasetParser"""

    def __init__(self, path: str, sampling: int = 1,
                 root_norm: bool = False):
        """Init

        Arguments:
            path {str} -- dataset path to be parser
            sampling {int} -- sampling factor

        Keywork Arguments:
            root_norm {bool} -- remove root translation and rotation
        """

        super().__init__()
        self.path = path
        self.sampling = sampling
        self.root_norm = root_norm

        self.cache_file = None
        self.cache = None
        self.indices = None

    @abstractmethod
    def index(self) -> list:
        """Index data

        Returns:
            list -- indices
        """
        raise NotImplementedError

    def get_element(self, idx: int) -> list:
        """Get element at given index

        Returns:
            list -- [path, idx]
        """

        if self.indices is None:
            self._logger.error("The dataset needs to be indexed first")

        idx_val = self.indices[idx]
        if isinstance(idx_val[0], str):
            return idx_val

        # convert encoded string and int with frame idx
        return [idx_val[0].decode('utf8'), int(idx_val[1])]

    @abstractmethod
    def process(self, file_path: str, fid: int):
        """Process single file

        Arguments:
            file_path {str} -- file path
            fid {int} -- frame id in the batch
        """
        raise NotImplementedError
