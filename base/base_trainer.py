# -*- coding: utf-8 -*-
"""
Base trainer class

@author: Denis Tome'

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__version__ = "0.2.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"

import os
import math
import shutil
from collections import OrderedDict
import torch
from base.base_dataset import DatasetInputFormat
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
                 reset, eval_epoch, val_log_step, dataset_input_type=None, **kwargs):
        """Init"""

        super().__init__(model, no_cuda)

        # ------------------- NN -------------------
        self.loss = loss
        self.metrics = metrics
        self.min_loss = math.inf
        self.reset = reset

        # ------------------- Hyper-params -------------------
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        # ------------------------- Log ------------------------
        self.global_step = 0
        self.train_log_step = train_log_step
        self.img_log_step = img_log_step
        self.val_log_step = val_log_step
        self.eval_epoch = eval_epoch
        self.model_version = model.version
        if dataset_input_type is None:
            self.is_aws = False
        else:
            self.is_aws = dataset_input_type == DatasetInputFormat.AWS.value

        # ---------------------- Generic -----------------------
        self.training_name = name
        self.start_epoch = 0
        self.start_iteration = 0
        self.training_info = None

        # --------------------- IO Related ---------------------
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = checkpoint_dir
        self.save_freq = save_freq
        self.ckpt_desc, self.ckpt_dir = self._update_name_convention(
            desc, desc_str)

        self.model_logger = ModelLogger(
            os.path.join(self.save_dir,
                         self.training_name,
                         self.ckpt_dir,
                         'log'),
            self.training_name)

        io.ensure_dir(
            os.path.join(self.save_dir,
                         self.training_name,
                         self.ckpt_dir))

        # ------------------- Resources -------------------
        if self.with_cuda:
            self.model.cuda()

        if self.is_multi_gpu():
            self._logger.info("Let's use %d GPUs!",
                              torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)

        # ------------------- Resume -------------------
        if resume:
            self._resume_checkpoint(resume)

    def _update_name_convention(self, desc: bool, desc_str: str):
        """Update model descriptor according to name convention

        Arguments:
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
                self.learning_rate, self.batch_size)

            if desc_str:
                ckpt_desc += '_{}'.format(desc_str)

            checkpoint_folder = '{}_{}'.format(
                self.model_version, ckpt_desc)
        else:
            ckpt_desc = ''
            checkpoint_folder = self.model_version

        return ckpt_desc, checkpoint_folder

    @staticmethod
    def get_dataset_len(data_loader) -> int:
        """Get dataset size

        Arguments:
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
        if self.with_cuda:
            self.model.cuda()
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._logger.info('Training epoch %d of %d',
                              epoch, self.epochs)
            epoch_loss = self._train_epoch(epoch)
            if self.eval_epoch:
                self._logger.info('Evaluating epoch %d of %d',
                                  epoch, self.epochs)
                epoch_val_loss, epoch_val_metrics = self._valid_epoch()
                self.model_logger.val.add_scalar('loss/iterations', epoch_val_loss,
                                                 self.global_step)
                for i, metric in enumerate(self.metrics):
                    metric.log_res(logger=self.model_logger.val,
                                   iter=self.global_step,
                                   error=epoch_val_metrics[i])
                    self._save_checkpoint(epoch, self.global_step, epoch_loss)

    def _dump_summary_info(self):
        """Save training summary"""

        def _summary_info():
            """Summary file describing the training
            hyper-parameters and logs

            Returns:
                dict: training log
            """

            return model_config

        if self.ckpt_desc != '':
            checkpoint_folder = '{}_{}'.format(
                self.model_version, self.ckpt_desc)
        else:
            checkpoint_folder = self.model_version

        info_file_path = os.path.join(self.save_dir,
                                      self.training_name,
                                      checkpoint_folder,
                                      'INFO.json')
        if not io.file_exists(info_file_path):
            info = _summary_info()
            io.write_json(info_file_path,
                          info)
        else:
            info = io.read_from_json(info_file_path)
        self.training_info = info

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, iteration: int, loss: float) -> str:
        """Save model

        Arguments:
            epoch (int): epoch number
            iteration (int): iteration number
            loss (float): loss value
        """

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

        if self.ckpt_desc != '':
            checkpoint_folder = '{}_{}'.format(
                self.model_version, self.ckpt_desc)
        else:
            checkpoint_folder = self.model_version

        filename = os.path.join(
            self.save_dir, self.training_name, checkpoint_folder,
            'ckpt_eph{:03d}_iter{:06d}_loss_{:.5f}.pth.tar'.format(
                epoch, iteration, loss))
        self._logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

        if loss == self.min_loss:
            shutil.copyfile(filename,
                            os.path.join(self.save_dir, self.training_name, checkpoint_folder,
                                         'model_best.pth.tar'))

        return filename

    def _resume_checkpoint(self, resume_path: str) -> None:
        """Resume model to be fine-tuned

        Arguments:
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
            if self.single_gpu:
                trained_dict = OrderedDict((k.replace('module.', ''), val)
                                           for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = OrderedDict(('module.{}'.format(k), val)
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

        self._logger.info("Checkpoint '%s' (epoch %d) loaded",
                          resume_path, self.start_epoch)
