# -*- coding: utf-8 -*-
"""
BaseModelExecution class to be inherithed by
those classes that involve training/testing the model

@author: Denis Tome'

"""
import os
from collections import OrderedDict
import torch
from torch.autograd import Variable
from base.template import FrameworkClass
from utils import io, is_model_parallel


class BaseModelExecution(FrameworkClass):
    """Base model execution class"""

    def __init__(self, model, no_cuda):

        super().__init__()

        self.model = model
        self.with_cuda = not no_cuda
        self.single_gpu = not self.is_multi_gpu()

        # ------------------- Resources -------------------
        if not torch.cuda.is_available():
            self.with_cuda = False

    @staticmethod
    def get_n_gpus() -> int:
        """Get number of GPUs

        Returns:
            int -- number of GPUs
        """

        return torch.cuda.device_count()

    def is_multi_gpu(self) -> bool:
        """Is multi-GPU available

        Returns:
            bool -- True if multi-GPU available
        """

        if self.with_cuda and (self.get_n_gpus() > 1):
            return True

        return False

    def _get_var(self, var):
        """Generate variable based on CUDA availability

        Arguments:
            var {undefined} -- variable to be converted

        Returns:
            tensor -- pytorch tensor
        """

        var = torch.FloatTensor(var)
        var = Variable(var)

        if self.with_cuda:
            var = var.cuda()

        return var

    def set_model_mode(self, model_mode):
        """Set model mode

        Arguments:
            model_mode {str/list} -- model mode
        """

        if isinstance(model_mode, list):
            model_mode = '{}_to_{}'.format(*model_mode)

        if self.single_gpu:
            self.model.set_model_mode(model_mode)
        else:
            self.model.module.set_model_mode(model_mode)

    def _resume_checkpoint(self, resume_path):
        """Resume model specified by the path

        Arguments:
            resume_path {str} -- path to directory containing the model
                                 or the model itself
        """

        if resume_path == 'init':
            self._logger.error('A model checkpoint needs to be provided!')

        # load model
        if not os.path.isfile(resume_path):
            resume_path = io.get_checkpoint(resume_path)

        self._logger.info("Loading checkpoint: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']

        if is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = OrderedDict((k.replace('module.', ''), val)
                                           for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = OrderedDict(('module.{}'.format(k), val)
                                           for k, val in checkpoint['state_dict'].items())

        self.model.load_state_dict(trained_dict)
        self._logger.info("Checkpoint '%s' loaded", resume_path)
