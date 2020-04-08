# -*- coding: utf-8 -*-
"""
Tester

@author: Denis Tome'

"""
import numpy as np
from tqdm import tqdm
from base import BaseTester
from model.modules import AvgPosesError
import utils

__all__ = [
    'Tester'
]


class Tester(BaseTester):
    """
    Tester class, inherited from BaseTester
    """

    def __init__(self, *args, **kwargs):
        """Init"""
        
        super().__init__(*args, metrics=None, **kwargs)
        self.metric = AvgPosesError()

    def save_res(self, pred, gt, info):
        """Save results

        Arguments:
            pred {numpy array} -- prediction
            gt {numpy array} -- ground truth
            info {dict} -- additional information about the frame
                           like location
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
        pbar = tqdm(self.data_loader)
        for (data, target, info) in pbar:
            data = self._get_var(data)
            target = self._get_var(target)

            # generete results
            output = self.model(data)

            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            err = self.save_res(output, target, info).tolist()
            if overall_error:
                overall_error.extend(err)
            else:
                overall_error = err

        self._logger.info('Overall error: %.3f', np.mean(overall_error))
        utils.write_h5('res.h5', overall_error)
