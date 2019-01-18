# -*- coding: utf-8 -*-
"""
Created on Jun 05 16:17 2018

@author: Denis Tome'

"""
import os
import utils
import torch
import logging
import numpy as np
import collections
from torch.autograd import Variable
from utils import ensure_dir, get_checkpoint, metadata_to_json


class BaseTester:
    """
    Base class for all dataset testers
    """

    def __init__(self, model, metrics, data_loader,
                 batch_size, save_dir, with_cuda,
                 model_path, verbosity,
                 output_name, verbosity_iter):

        self._logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.test_data_loader = data_loader
        self.batch_size = batch_size
        self.metrics = metrics
        self.output_name = output_name
        self.save_dir = utils.abs_path(save_dir)
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
                self._logger.info('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
            self.single_gpu = False

        ensure_dir(os.path.join(save_dir, self.output_name))
        if model_path:
            self._resume_checkpoint(model_path)

    def _update_exec_time(self, t):
        self.exec_time += t / self.batch_size

    def _save_testing_info(self, metrics):
        file_path = os.path.join(self.save_dir,
                                 self.output_name,
                                 'TEST.json')
        num_elems = len(self.test_data_loader)

        info = {
            'num_batches': num_elems,
            'batch_size': self.batch_size,
            'num_frames': num_elems * self.batch_size,
            'model': self.model._get_name(),
            'avg_exec_time': self.exec_time / num_elems,
        }

        for mid in np.arange(len(self.metrics)):
            info[self.metrics[mid].__class__.__name__] = metrics[mid]/len(self.test_data_loader)

        # save json file
        metadata_to_json(file_path=file_path,
                         info=info)

    def _get_var(self, var):
        """
        Generate variable to be used by cuda
        :param var
        :return var to be used by cuda
        """
        var = torch.FloatTensor(var)
        var = Variable(var)

        if self.with_cuda:
            var = var.cuda()

        return var

    def predict_poses(self):
        raise NotImplementedError

    def test(self):
        self.predict_poses()

    def _resume_checkpoint(self, resume_path, epoch=None, iteration=None):
        """
        Resume model saved in resume_dir at the specified epoch and iteration

        :param resume_path: path to the model or the dir with the models
        :param epoch: epoch of the model
        :param iteration: iteration of the model
        """

        # load model
        if not os.path.isfile(resume_path):
            resume_path = get_checkpoint(resume_path, epoch, iteration)

        self._logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']

        if utils.is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = collections.OrderedDict((k.replace('module.', ''), val)
                                                       for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = collections.OrderedDict(('module.{}'.format(k), val)
                                                       for k, val in checkpoint['state_dict'].items())

        self.model.load_state_dict(trained_dict)
        self._logger.info("Checkpoint '{}' loaded".format(resume_path))
