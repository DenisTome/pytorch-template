# -*- coding: utf-8 -*-
"""
Base code test to make sure a specific function
is working as expected

@author: Denis Tome'

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"

from abc import abstractmethod
from logger import TestLogger


class BaseCodeTest():
    """BaseCodeTest"""

    def __init__(self):
        super().__init__()
        self._logger = TestLogger(self.__class__.__name__)

        self.eps = 1e-5

    def passed(self):
        """Test passed"""
        self._logger.test_passed()

    def failed(self):
        """Test failed"""
        self._logger.test_failed()
        raise TestException

    @abstractmethod
    def test(self) -> None:
        """Test"""
        raise NotImplementedError

    def __call__(self):
        """Run test"""
        self.test()


class TestException(Exception):
    """Raised when test fails"""
    pass
