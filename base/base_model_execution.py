# -*- coding: utf-8 -*-
"""
BaseModelExecution class to be inherithed by
those classes that involve training/testing the model

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"

import os
from collections import OrderedDict
import torch
from torch.autograd import Variable
from base.template import FrameworkClass
import utils.io as io
from utils.util import is_model_parallel


class BaseModelExecution(FrameworkClass):
    """Base model execution class"""

    def __init__(self, model, no_cuda: bool):
        """Init

        Args:
            model (nn.Module): model
            no_cuda (bool): with cuda
        """

        super().__init__()

        self._model = model
        self._model_name = self._model.name
        self._model_version = self._model.version
        self._with_cuda = not no_cuda
        self._single_gpu = not self.is_multi_gpu()

        # ------------------- Resources -------------------

        if not torch.cuda.is_available():
            self._with_cuda = False

    @staticmethod
    def get_n_gpus() -> int:
        """Get number of GPUs

        Returns:
            int: number of GPUs
        """

        return torch.cuda.device_count()

    @property
    def is_multi_gpu(self) -> bool:
        """Is multi-GPU available for model

        Returns:
            bool: True if multi-GPU available
        """

        if self._with_cuda and (self.get_n_gpus() > 1):
            return True

        return False

    def _get_var(self, var: torch.Tensor) -> torch.autograd.Variable:
        """Generate variable based on CUDA availability

        Args:
            var (torch.Tensor): tensor to be turned into a variable

        Returns:
            torch.autograd.Variable: pytorch tensor
        """

        var = torch.FloatTensor(var)
        var = Variable(var)

        if self._with_cuda:
            var = var.cuda()

        return var

    def _resume_checkpoint(self, resume_path: str):
        """Resume model specified by the path

        Args:
            resume_path (str): path to directory containing the model
                               or the model itself. If it's `init` than the model
                               is randomly initialized.
        """

        if resume_path is None:
            AssertionError('resume path cannot be None!')

        if resume_path == 'init':
            return

        # ------------------- load model -------------------

        if not os.path.isfile(resume_path):
            resume_path = io.get_checkpoint(resume_path)

        self._logger.info("Loading checkpoint: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']

        if is_model_parallel(checkpoint):
            if self._single_gpu:
                trained_dict = OrderedDict((k.replace('module.', ''), val)
                                           for k, val in checkpoint['state_dict'].items())
        else:
            if not self._single_gpu:
                trained_dict = OrderedDict(('module.{}'.format(k), val)
                                           for k, val in checkpoint['state_dict'].items())

        self._model.load_state_dict(trained_dict)
        self._logger.info("Checkpoint '%s' loaded", resume_path)
