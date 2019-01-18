# -*- coding: utf-8 -*-
"""
Created on Jun 25 15:56 2018

@author: Denis Tome'

"""
from base.template import FrameworkClass

__all__ = [
    'BaseMetric'
]


class BaseMetric(FrameworkClass):

    def __init__(self):
        """Create class"""
        super().__init__()
        self.metric_init = 0.0
        self.name_format = self.__class__.__name__

    def eval(self, pred, gt):
        """
        Compute metric for a specific sample
        :param pred: predictions
        :param gt: ground truth
        :param scale: scaling factor
        :return: metric value (scala, list, etc. depending on the metric)
        """
        raise NotImplementedError('Abstract method in BaseMetric class...')

    def add_results(self, res, pred, gt):
        """
        Update results
        :param res: metric computed ad previous step
        :param pred: predictions
        :param gt: ground truth
        :return: updated metric value (scala, list, etc. depending on the metric)
        """
        if res is None:
            res = self.metric_init

        return res + self.eval(pred, gt)

    def log(self, logger, iter, inputs, targets=None):
        """
        Compute metric and add to the log file
        """
        error = self.eval(inputs, targets)
        self.log_res(logger,
                     iter,
                     error)

    def log_res(self, logger, iter, error):
        logger.add_scalar('metrics/{0}'.format(self.name_format),
                          error,
                          iter)

    def _desc(self, **kwargs):
        """
        Return short string indicating description of the metric
        """
        raise NotImplementedError('Abstract method in BaseMetric class...')
