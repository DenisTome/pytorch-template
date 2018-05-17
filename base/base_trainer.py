import os
import re
import math
import shutil
import torch
import logging
from logger.logger import Logger
from utils.util import ensure_dir


class BaseTrainer:
    """ Base class for all trainers.

    Note:
        Modify if you need to change logging style, checkpoint naming, or something else.
    """

    def __init__(self, model, loss, metrics, optimizer, epochs,
                 training_name, save_dir, save_freq, with_cuda,
                 resume, verbosity, train_log_step, verbosity_iter):

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
        self.start_epoch = 1
        self.start_iteration = 0
        self.model_logger = Logger(os.path.join(save_dir,
                                                self.training_name,
                                                'log'),
                                   self.training_name)

        # check that we can run on GPU
        if not torch.cuda.is_available():
            self.with_cuda = False

        if self.with_cuda and (torch.cuda.device_count() > 1):
            if self.verbosity:
                self._logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        ensure_dir(os.path.join(save_dir,
                                self.training_name))
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.verbosity:
                self._logger.info('Training epoch {:d} of {:d}'.format(epoch,
                                                                       self.epochs))
            epoch_loss, max_iter = self._train_epoch(epoch)
            self._save_checkpoint(epoch, max_iter, epoch_loss)
            self.start_iteration = 0

    def _train_epoch(self, epoch):
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
        filename = os.path.join(self.save_dir,
                                self.training_name,
                                'ckpt_eph{:02d}_iter{:06d}_loss_{:.5f}.pth.tar'.format(epoch,
                                                                                       iteration,
                                                                                       loss))
        self._logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename, os.path.join(self.save_dir,
                                                   self.training_name,
                                                   'model_best.pth.tar'))

    def _resume_checkpoint(self, resume_dir, epoch, iteration):
        """
        Resume model saved in resume_dir at the specified epoch and iteration

        :param resume_dir: directory with the models
        :param epoch: epoch of the model
        :param iteration: iteration of the model
        """

        models_list = [f for f in os.listdir(resume_dir) if f.endswith(".pth.tar")]

        # getting the right model
        r = re.compile("ckpt_eph{:02d}_iter{:06d}_.*".format(epoch, iteration))
        model_name = [m.group(0) for l in models_list for m in [r.search(l)] if m]

        if not model_name:
            self._logger.error('Model {}/ckpt_eph{:02d}_iter{:06d} does not exist'.format(resume_dir,
                                                                                          epoch,
                                                                                          iteration))
        # load model
        resume_path = os.path.join(resume_dir,
                                   model_name[0])
        self._logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['iter']
        self.min_loss = checkpoint['min_loss']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
