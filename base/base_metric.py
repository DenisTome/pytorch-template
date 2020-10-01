# -*- coding: utf-8 -*-
"""
Base metric class to be extended

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"

from abc import abstractmethod
import numpy as np
from base.template import FrameworkClass
from logger.console_logger import ConsoleLogger


class BaseMetric(FrameworkClass):
    """Base Metric class"""

    def __init__(self, logger: ConsoleLogger):
        """Init

        Args:
            logger (ConsoleLogger): train logger where to save results to.
        """

        super().__init__()
        self._metric_init = 0.0
        self._logger = logger

    @abstractmethod
    def evaluate(self, pred: np.array, gt: np.array) -> float:
        """Compute metric

        Args:
            pred (np.array): predictions
            gt (np.array): ground truth

        Returns:
            float: result of evaluation
        """
        raise NotImplementedError

    def eval_and_log(self, pred: np.array, gt: np.array, iteration: int) -> None:
        """Compute error and save it in the log

        Args:
            pred (np.array): prediction
            gt (np.array): ground truth
            iteration (int): iteration number
        """

        error = self.evaluate(pred, gt)
        self.log_res(error, iteration)

    def log_res(self, error: float, iteration: int) -> None:
        """Add given error to log file

        Args:
            error (float): error
            iteration (int): iteration number
        """

        self._logger.add_scalar(
            'metrics/{0}'.format(self._name_format),
            error, iteration)

    @abstractmethod
    @property
    def desc(self):
        """Get description of the metric"""
        return 'BaseMetric'
