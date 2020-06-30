# -*- coding: utf-8 -*-
"""
Base tester class to be extended

@author: Denis Tome'

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__version__ = "0.1.2"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"

import os
import re
import torch
import numpy as np
from base import BaseModelExecution
import utils.io as io


class BaseTester(BaseModelExecution):
    """
    Base class for all dataset testers
    """

    def __init__(self, model, metrics, test_loader, batch_size,
                 output, desc, no_cuda, resume, name, dataset,
                 **kwargs):
        """Init"""

        super().__init__(model, no_cuda)

        # ------------------- NN -------------------
        self.metrics = metrics
        self.min_loss = np.inf

        # ------------------- Hyper-params -------------------
        self.batch_size = batch_size

        # ------------------- Log -------------------
        self.exec_time = 0
        self.dataset = dataset

        # ------------------- IO Related -------------------
        self.test_loader = test_loader
        self.output_name = name
        self.save_dir = io.ensure_dir(io.abs_path(output))
        self.model_resume_path = resume
        io.ensure_dir(os.path.join(self.save_dir, self.output_name))

        if desc:
            learning_rate = self._get_learning_rate(resume)
            self.desc = 'lr_{}_b_{}'.format(
                learning_rate, self.batch_size
            )
            eph_num = self._get_epoch_number(resume)
            if eph_num:
                self.desc = '{}_ckpt_eph_{}'.format(self.desc, eph_num)
        else:
            self.desc = ''

        # ------------------- Resources -------------------
        if self.with_cuda:
            self.model.cuda()

        if self.is_multi_gpu():
            self._logger.info('Let\'s use %d GPUs!',
                              torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)

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

    def _update_exec_time(self, t):
        """Update execution time

        Arguments:
            t (long): execution time in ms
        """

        self.exec_time += t / self.batch_size

    def _save_testing_info(self, metrics):
        """Save test information

        Arguments:
            metrics (Metric): metrics used for evaluation
        """

        file_path = os.path.join(self.save_dir,
                                 self.output_name,
                                 'TEST.json')
        num_elems = len(self.test_loader)

        info = {
            'num_batches': num_elems,
            'batch_size': self.batch_size,
            'num_frames': num_elems * self.batch_size,
            'model': self.model.name,
            'avg_exec_time': self.exec_time / num_elems,
        }

        for mid in np.arange(len(self.metrics)):
            info[self.metrics[mid].__class__.__name__] = metrics[mid] / \
                len(self.test_loader)

        # save json file
        io.write_json(file_path, info)

    def test(self):
        """Run test on the test-set"""
        raise NotImplementedError()
