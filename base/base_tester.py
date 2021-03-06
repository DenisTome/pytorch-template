# -*- coding: utf-8 -*-
"""
Base tester class to be extended

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"

from abc import abstractmethod
import os
import re
import time
import torch
import numpy as np
from base import BaseModelExecution
import utils.io as io


class BaseModelEval(BaseModelExecution):
    """
    Base class for all dataset testers
    """

    def __init__(self, model, metrics, test_loader, batch_size,
                 output, desc, no_cuda, resume, name, dataset,
                 **kwargs):
        """Init"""

        super().__init__(model, no_cuda)

        # ------------------- NN -------------------

        self._metrics = metrics
        self._min_loss = np.inf

        # ------------------- Hyper-params -------------------

        self._batch_size = batch_size

        # ------------------- Log -------------------

        self._exec_time = 0
        self._dataset = dataset
        self._time_start = 0
        self._metric_results = None

        # ------------------- IO Related -------------------

        self._test_loader = test_loader
        self._output_name = name
        self._save_dir = io.ensure_dir(io.abs_path(output))
        self._model_resume_path = resume
        io.ensure_dir(os.path.join(self._save_dir, self._output_name))

        if desc:
            # add defscription to the checkpoint file to differentiante
            # between several configurations of the same architecture
            learning_rate = self._get_learning_rate(resume)
            self.desc = 'lr_{}_b_{}'.format(
                learning_rate, self._batch_size)

            eph_num = self._get_epoch_number(resume)
            if eph_num:
                self.desc = '{}_ckpt_eph_{}'.format(self.desc, eph_num)
        else:
            self.desc = ''

        # ------------------- Resources -------------------

        if self._with_cuda:
            self._model.cuda()

        if self.is_multi_gpu():
            self._logger.info('Let\'s use %d GPUs!',
                              torch.cuda.device_count())
            self._model = torch.nn.DataParallel(self._model)

        # ------------------- Resume -------------------

        self._resume_checkpoint(resume)

    @staticmethod
    def _get_learning_rate(path: str) -> str:
        """Get learning rate from model path

        Arguments:
            path (str): model path

        Returns:
            str: learning rate
        """

        lr = re.findall(r'_lr_(\d*\.\d+)',
                        path)[0]
        return lr

    @staticmethod
    def _get_epoch_number(path: str) -> str:
        """Get checkpoint epoch number

        Arguments:
            path (str): model path

        Returns:
            str: epoch number
        """

        ckpt = re.findall(r'ckpt_eph(\d*)', path)
        if ckpt:
            return ckpt[0]

        return ''

    @staticmethod
    def _get_model_version(path: str) -> str:
        """Get model version

        Args:
            path (str): path

        Returns:
            str: version
        """

        version = re.findall(r'(v\d+\.\d+\.\d+(_lr_(\d*\.\d+)_b_(\d+)_e_(\d+))?)',
                             path)[0][0]
        return version

    def _start_time(self):
        """Get time"""
        self._time_start = time.time()

    def _stop_time(self):
        """Stop and update total execution time"""

        stop_time = time.time()
        time_int = stop_time - self._start_time
        self._exec_time += time_int / self._batch_size

    def _eval_on_metrices(self, pred: np.array, gt: np.array) -> float:

        results = np.zeros(len(self._metrics), dtype=float)
        for mid, metric in enumerate(self._metrics):
            results[mid] = metric.evaluate(pred, gt)

        if self._metric_results is None:
            self._metric_results = [results]
            return

        self._metric_results.append(results)

    def _save_testing_info(self):
        """Save test information"""

        file_path = os.path.join(self._save_dir,
                                 self._output_name,
                                 'TEST.json')
        num_elems = len(self._test_loader)

        info = {
            'num_batches': num_elems,
            'batch_size': self._batch_size,
            'num_frames': num_elems * self._batch_size,
            'model': self._model_name,
            'avg_exec_time': self._exec_time / num_elems,
        }

        results = np.array(self._metric_results)
        for mid in np.arange(len(self._metrics)):
            info[self._metrics[mid].__class__.__name__] = np.mean(results[:, mid])

        # save json file
        io.write_json(file_path, info)

    @abstractmethod
    def test(self):
        """Run test on the test-set"""
        raise NotImplementedError
