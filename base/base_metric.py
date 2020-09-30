# -*- coding: utf-8 -*-
"""
Base metric class to be extended

@author: Denis Tome'

"""

__version__ = "0.1.1"

from base.template import FrameworkClass


class BaseMetric(FrameworkClass):
    """Base Metric class"""

    def __init__(self):
        """Initialize class"""

        super().__init__()
        self.metric_init = 0.0
        self.name_format = self._desc

    def eval(self, pred, gt):
        """Compute metric

        Arguments:
            pred (numpy array): predicted pose
            gt (numpy array): ground truth pose
        """
        raise NotImplementedError('Abstract method in BaseMetric class...')

    def add_results(self, res, pred, gt):
        """Update results

        Arguments:
            res (float): sum of past evaluations
            pred (numpy array): predicted pose
            gt (numpy array): ground truth pose

        Returns:
            float: sum of evaluations
        """

        if res is None:
            res = self.metric_init

        return res + self.eval(pred, gt)

    def log(self, logger, iteration, pred, gt, dataset=None):
        """Evaluate and add it to the log file

        Arguments:
            logger (Logger): class responsible for logging info
            iteration (int): iteration number
            pred (numpy array): prediction
            gt (numpy array): ground truth
            dataset (str): dataset name
        """

        error = self.eval(pred, gt)
        self.log_res(logger,
                     iteration,
                     error,
                     dataset)

    def log_res(self, logger, iteration, error, dataset=None):
        """Add result to log file

        Arguments:
            logger (Logger): class responsible for logging into
            iteration (int): iteration number
            error (float): error value
            dataset (str): dataset name
        """

        if dataset:
            logger.add_scalar('metrics/{0}_{1}'.format(dataset, self.name_format),
                              error,
                              iteration)
        else:
            logger.add_scalar('metrics/{0}'.format(self.name_format),
                              error,
                              iteration)

    @property
    def _desc(self):
        """Name of the descriptor to use in thensorboard
        to represent the metric"""

        raise NotImplementedError('Abstract method in BaseMetric class...')
