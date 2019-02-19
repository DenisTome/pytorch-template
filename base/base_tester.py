# -*- coding: utf-8 -*-
"""
Base tester class to be extended

@author: Denis Tome'

"""
import os
import collections
import torch
from torch.autograd import Variable
import numpy as np
import utils.io as io
from utils import is_model_parallel
from base.template import FrameworkClass


class BaseTester(FrameworkClass):
    """
    Base class for all dataset testers
    """

    def __init__(self, model, metrics, data_loader,
                 batch_size, save_dir, with_cuda,
                 model_path, verbosity, output_name,
                 verbosity_iter):

        super().__init__()

        self.model = model
        self.test_data_loader = data_loader
        self.batch_size = batch_size
        self.metrics = metrics
        self.output_name = output_name
        self.save_dir = io.abs_path(save_dir)
        self.with_cuda = with_cuda
        self.verbosity = verbosity
        self.verbosity_iter = verbosity_iter
        self.min_loss = np.inf
        self.exec_time = 0
        self.single_gpu = True

        # check that we can run on GPU
        if not torch.cuda.is_available():
            self.with_cuda = False

        if self.with_cuda and (torch.cuda.device_count() > 1):
            if self.verbosity:
                self._logger.info('Let\'s use %d GPUs!',
                                  torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            self.single_gpu = False

        io.ensure_dir(os.path.join(save_dir, self.output_name))
        if model_path:
            self._resume_checkpoint(model_path)

    def _update_exec_time(self, t):
        """Update execution time

        Arguments:
            t {long} -- execution time in ms
        """

        self.exec_time += t / self.batch_size

    def _save_testing_info(self, metrics):
        """Save test information

        Arguments:
            metrics {Metric} -- metrics used for evaluation
        """

        file_path = os.path.join(self.save_dir,
                                 self.output_name,
                                 'TEST.json')
        num_elems = len(self.test_data_loader)

        info = {
            'num_batches': num_elems,
            'batch_size': self.batch_size,
            'num_frames': num_elems * self.batch_size,
            'model': self.model.name,
            'avg_exec_time': self.exec_time / num_elems,
        }

        for mid in np.arange(len(self.metrics)):
            info[self.metrics[mid].__class__.__name__] = metrics[mid] / \
                len(self.test_data_loader)

        # save json file
        io.write_json(file_path, info)

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

    def test(self):
        """Run test on the test-set"""
        raise NotImplementedError()

    def _resume_checkpoint(self, path):
        """Resume model specified by the path

        Arguments:
            resume_path {str} -- path to directory containing the model
                                 or the model itself
        """

        # load model
        if not os.path.isfile(resume_path):
            resume_path = io.get_checkpoint(path)

        self._logger.info("Loading checkpoint: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']

        if is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = collections.OrderedDict((k.replace('module.', ''), val)
                                                       for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = collections.OrderedDict(('module.{}'.format(k), val)
                                                       for k, val in checkpoint['state_dict'].items())

        self.model.load_state_dict(trained_dict)
        self._logger.info("Checkpoint '%s' loaded", resume_path)
