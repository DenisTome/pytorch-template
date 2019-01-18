# -*- coding: utf-8 -*-
"""
Created on Jun 05 16:17 2018

@author: Denis Tome'

"""
import os
import utils
import numpy as np
from base.base_tester import BaseTester
from model.modules.metric import AvgPosesError

__all__ = [
    'Tester'
]


class Tester(BaseTester):
    """
    Tester class, inherited from BaseTester
    """

    def __init__(self, model, data_loader,
                 batch_size, save_dir, resume,
                 verbosity, verbosity_iter,
                 with_cuda, output_name):
        super().__init__(model, None, data_loader,
                         batch_size, save_dir, with_cuda,
                         resume, verbosity,
                         output_name, verbosity_iter)
        self.metric = AvgPosesError()

    def save_res(self, pred, gt, info):
        """
        Save results in files
        :param pred
        :param gt
        :param info: dictionary containing all info about the batch
                     like path, file_name, etc.
        """
        raise NotImplementedError()

    def test(self):
        """
        Save predicted 2D joints in output dir along with
        some additional information
        """
        if self.verbosity:
            self._logger.info('Predicting 3D poses on test-set')

        self.model.eval()
        if self.with_cuda:
            self.model.cuda()

        overall_error = None
        for bid, (data, target, info) in enumerate(self.test_data_loader):
            data = self._get_var(data)
            target = self._get_var(target)

            if (bid % self.verbosity_iter == 0) & (self.verbosity == 2):
                self._logger.info('Test, batch {:d}/{:d}'.format(
                    bid, len(self.test_data_loader)))
                if overall_error is not None:
                    self._logger.info('Error: {:.3f}'.format(np.mean(overall_error)))

            output = self.model(data)

            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            err = self.save_res(output, target, info).tolist()
            if overall_error:
                overall_error.extend(err)
            else:
                overall_error = err

        self._logger.info('Overall error: {}'.format(np.mean(overall_error)))
        utils.write_h5('res.h5', overall_error)