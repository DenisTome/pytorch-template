# -*- coding: utf-8 -*-
"""
ConsoleLogger class to print data in
the console.

@author: Denis Tome'

"""
from enum import Enum
import logging

__all__ = [
    'TestLogger'
]


class ResultType(Enum):
    """Test result type"""

    PASS = "passed"
    FAIL = "failed"


class CustomFormatter(logging.Formatter):
    """Custom formatter"""

    TEXT = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        """initializer"""

        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"

        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        """format message"""

        color = self.WHITE
        if record.msg == ResultType.PASS.value:
            color = self.GREEN

        if record.msg == ResultType.FAIL.value:
            color = self.RED

        if record.levelno == logging.WARN:
            color = self.WARNING

        self._style._fmt = "{}%(asctime)s %(name)s{}: %(message)s{}".format(
            self.TEXT, color, self.WHITE)

        return logging.Formatter.format(self, record)


class TestLogger():
    """Console logger"""

    def __init__(self, name='main'):
        super().__init__()

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)

        self._logger.addHandler(console_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """info"""
        self._logger.warning(*args, **kwargs)

    def test_passed(self):
        """info"""
        self._logger.info(msg=ResultType.PASS.value)

    def test_failed(self):
        """info"""
        self._logger.info(msg=ResultType.FAIL.value)
