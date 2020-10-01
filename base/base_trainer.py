# -*- coding: utf-8 -*-
"""
Base trainer class

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"

import os
import math
import shutil
from collections import OrderedDict
import torch
from base import BaseModelExecution
from logger.model_logger import ModelLogger
import utils.io as io
from utils.config import model_config
from utils.util import is_model_parallel


class BaseTrainer(BaseModelExecution):
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, train_loader, val_loader,
                 batch_size, learning_rate, epochs, name, checkpoint_dir, save_freq,
                 no_cuda, resume, img_log_step, train_log_step, desc, desc_str,
                 reset, eval_epoch, val_log_step, **kwargs):
        """Init"""

        super().__init__(model, no_cuda)

        # ------------------- NN -------------------

        self._loss = loss
        self._metrics = metrics
        self._min_loss = math.inf
        self._reset = reset

        # ------------------- Hyper-params -------------------

        self._lr = learning_rate
        self._opt = optimizer
        self._epochs = epochs
        self._bs = batch_size

        # ------------------------- Log ------------------------

        self._global_step = 0
        self._train_log_step = train_log_step
        self._img_log_step = img_log_step
        self._val_log_step = val_log_step
        self._eval_epoch = eval_epoch

        # ---------------------- Generic -----------------------

        self._training_name = name
        self._start_epoch = 0
        self._start_iteration = 0
        self._training_info = None

        # --------------------- IO Related ---------------------

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._save_dir = checkpoint_dir
        self._save_freq = save_freq
        self._ckpt_desc, self._ckpt_dir = self._update_name_convention(
            desc, desc_str)

        self._model_logger = ModelLogger(
            os.path.join(
                self._save_dir, self._training_name, self._ckpt_dir, 'log'),
            self._training_name)

        io.ensure_dir(
            os.path.join(self._save_dir,
                         self._training_name,
                         self._ckpt_dir))

        # ------------------- Resources -------------------

        if self._with_cuda:
            self._model.cuda()

        if self.is_multi_gpu():
            self._logger.info("Let's use %d GPUs!",
                              torch.cuda.device_count())
            self._model = torch.nn.DataParallel(self._model)

        # ------------------- Resume -------------------

        if resume:
            self._resume_checkpoint(resume)

    def _update_name_convention(self, desc: bool, desc_str: str):
        """Update model descriptor according to name convention

        Args:
            desc (bool): add descriptor to name
            desc_str (str): str to attach

        Returns:
            str: descriptor
            str: checkpoint dir
        """

        if desc_str:
            desc = True

        if desc:
            ckpt_desc = 'lr_{}_b_{}'.format(
                self._lr, self._bs)

            if desc_str:
                ckpt_desc += '_{}'.format(desc_str)

            checkpoint_folder = '{}_{}'.format(
                self._model_version, ckpt_desc)
        else:
            ckpt_desc = ''
            checkpoint_folder = self._model_version

        return ckpt_desc, checkpoint_folder

    @staticmethod
    def get_dataset_len(data_loader) -> int:
        """Get dataset size

        Args:
            data_loader (DataLoader): dataset loader

        Returns:
            int: dataset length
        """

        try:
            size = len(data_loader)
        except TypeError:
            size = data_loader.batch_sampler.n_elems

        return size

    def train(self):
        """Train model"""

        self._dump_summary_info()
        for epoch in range(self.start_epoch, self._epochs + 1):
            self._logger.info('Training epoch %d of %d',
                              epoch, self._epochs)
            epoch_loss = self._train_epoch(epoch)

            if self._eval_epoch:
                self._logger.info('Evaluating epoch %d of %d',
                                  epoch, self._epochs)
                epoch_val_loss, epoch_val_metrics = self._valid_epoch()
                self._model_logger.val.add_scalar('loss/iterations', epoch_val_loss,
                                                  self._global_step)

                for i, metric in enumerate(self._metrics):
                    metric.log_res(logger=self._model_logger.val,
                                   iter=self._global_step,
                                   error=epoch_val_metrics[i])
                    self._save_checkpoint(epoch, self._global_step, epoch_loss)

    def _dump_summary_info(self):
        """Save training summary"""

        def _summary_info():
            """Summary file describing the training
            hyper-parameters and logs

            Returns:
                dict: training log
            """

            return model_config

        if self._ckpt_desc != '':
            checkpoint_folder = '{}_{}'.format(
                self._model_version, self._ckpt_desc)
        else:
            checkpoint_folder = self._model_version

        info_file_path = os.path.join(self._save_dir,
                                      self._training_name,
                                      checkpoint_folder,
                                      'INFO.json')
        if not io.file_exists(info_file_path):
            info = _summary_info()
            io.write_json(info_file_path,
                          info)
        else:
            info = io.read_from_json(info_file_path)
        self._training_info = info

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, iteration: int, loss: float) -> str:
        """Save model

        Args:
            epoch (int): epoch number
            iteration (int): iteration number
            loss (float): loss value
        """

        if loss < self._min_loss:
            self._min_loss = loss

        arch = type(self._model).__name__
        state = {
            'epoch': epoch,
            'iter': iteration,
            'arch': arch,
            'state_dict': self._model.state_dict(),
            'optimizer': self._opt.state_dict(),
            'min_loss': self._min_loss,
        }

        if self._ckpt_desc != '':
            checkpoint_folder = '{}_{}'.format(
                self._model_version, self._ckpt_desc)
        else:
            checkpoint_folder = self._model_version

        filename = os.path.join(
            self._save_dir, self._training_name, checkpoint_folder,
            'ckpt_eph{:03d}_iter{:06d}_loss_{:.5f}.pth.tar'.format(
                epoch, iteration, loss))
        self._logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

        if loss == self._min_loss:
            shutil.copyfile(filename,
                            os.path.join(self._save_dir, self._training_name, checkpoint_folder,
                                         'model_best.pth.tar'))

        return filename

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Resume model to be fine-tuned

        Args:
            resume_path (str): path to the directory or model to be resumed
        """

        if resume_path == 'init':
            return

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

        if not self._reset:
            self.start_iteration = checkpoint['iter'] + 1
            self.start_epoch = checkpoint['epoch']
            try:
                self._global_step = checkpoint['global_step'] + 1
            except KeyError:
                self._global_step = self.start_iteration
            self._min_loss = checkpoint['min_loss']
            self._opt.load_state_dict(checkpoint['optimizer'])

        self._logger.info("Checkpoint '%s' (epoch %d) loaded",
                          resume_path, self.start_epoch)
