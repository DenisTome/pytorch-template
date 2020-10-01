# -*- coding: utf-8 -*-
"""
Tester

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
from tqdm import tqdm
from base import BaseModelEval

__all__ = [
    'ModelEval'
]


class ModelEval(BaseModelEval):
    """
    Tester class, inherited from BaseTester
    """

    def test(self):
        """Save predicted 2D joints in output dir along with
        some additional information"""

        self._logger.info('Running model on test-set')

        self._model.eval()

        pbar = tqdm(self._test_loader)
        for (data, target, _) in pbar:

            data = self._get_var(data)
            target = self._get_var(target)

            # generate results
            self._start_time()
            output = self._model(data)
            self._stop_time()

            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            self._eval_on_metrices(output, target)

        self._logger.info('Save results...')
        self._save_testing_info()
        self._logger.info('Done.')
