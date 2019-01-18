# -*- coding: utf-8 -*-
"""
Created on Jun 05 16:17 2018

@author: Denis Tome'

Base trainer class.

"""
import os
import math
import shutil
import torch
import utils
import logging
import collections
import numpy as np
from logger.logger import Logger
from torch.autograd import Variable


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, epochs, training_name,
                 save_dir, save_freq, with_cuda, resume, verbosity,
                 train_log_step, verbosity_iter, reset=False, eval_epoch=False):

        self._logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.epochs = epochs
        self.training_name = training_name
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.with_cuda = with_cuda
        self.verbosity = verbosity
        self.verbosity_iter = verbosity_iter
        self.train_log_step = train_log_step
        self.min_loss = math.inf
        self.start_epoch = 0
        self.start_iteration = 0
        self.model_logger = Logger(
            os.path.join(save_dir, self.training_name, 'log'),
            self.training_name)
        self.training_info = None
        self.eval_epoch = eval_epoch
        self.reset = reset
        self.single_gpu = True

        # check that we can run on GPU
        if not torch.cuda.is_available():
            self.with_cuda = False

        if self.with_cuda and (torch.cuda.device_count() > 1):
            if self.verbosity:
                self._logger.info("Let's use {} GPUs!".format(
                    torch.cuda.device_count())
                )
            self.single_gpu = False
            self.model = torch.nn.DataParallel(self.model)

        utils.ensure_dir(os.path.join(save_dir, self.training_name))
        if resume:
            self._resume_checkpoint(resume)

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

    def train(self):
        self._dump_summary_info()
        if self.with_cuda:
            self.model.cuda()
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.verbosity:
                self._logger.info('Training epoch {:d} of {:d}'.format(
                    epoch, self.epochs))
            epoch_loss, max_iter = self._train_epoch(epoch)
            if self.eval_epoch:
                self._logger.info('Evaluating epoch {:d} of {:d}'.format(
                    epoch, self.epochs))
                epoch_val_loss, epoch_val_metrics = self._valid_epoch()
                self.model_logger.val.add_scalar('loss/iterations', epoch_val_loss,
                                                 self.global_step)
                for i, metric in enumerate(self.metrics):
                    metric.log_res(logger=self.model_logger.val,
                                   iter=self.global_step,
                                   error=epoch_val_metrics[i])
                    self._save_checkpoint(epoch, self.global_step, epoch_loss)

    def _dump_summary_info(self):
        info_file_path = os.path.join(self.save_dir,
                                      self.training_name,
                                      'INFO.json')
        if not utils.file_exists(info_file_path):
            info = self._summary_info()
            utils.write_json(info_file_path,
                             info)
        else:
            info = utils.read_from_json(info_file_path)
        self.training_info = info

    def _update_summary(self, global_step, loss, metrics):
        self.training_info['global_step'] = global_step
        self.training_info['val_loss'] = loss
        for idx, metric in enumerate(self.metrics):
            m = metrics[idx]
            if isinstance(m, np.ndarray):
                m = m.tolist()
            self.training_info['val_{}'.format(metric._desc)] = m

        info_file_path = os.path.join(self.save_dir,
                                      self.training_name,
                                      'INFO.json')
        utils.write_json(info_file_path,
                         self.training_info)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self):
        raise NotImplementedError

    def _summary_info(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, iteration, loss):
        if loss < self.min_loss:
            self.min_loss = loss
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'iter': iteration,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'min_loss': self.min_loss,
        }
        filename = os.path.join(
            self.save_dir, self.training_name,
            'ckpt_eph{:02d}_iter{:06d}_loss_{:.5f}.pth.tar'.format(
                epoch, iteration, loss))
        self._logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename,
                            os.path.join(self.save_dir, self.training_name,
                                         'model_best.pth.tar'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume model saved in resume_path or if it's a directory
        the last model contained in it/

        :param resume_path: path to file or dir with the models
        """
        if not os.path.isfile(resume_path):
            resume_path = utils.get_checkpoint(resume_path)

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

        if not self.reset:
            self.start_iteration = checkpoint['iter'] + 1
            self.start_epoch = checkpoint['epoch']
            try:
                self.global_step = checkpoint['global_step'] + 1
            except KeyError:
                self.global_step = self.start_iteration
            self.min_loss = checkpoint['min_loss']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self._logger.info("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))
