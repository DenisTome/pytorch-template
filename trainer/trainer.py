import numpy as np
import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self, model, loss, metrics, data_loader,
                 optimizer, epochs, training_name, save_dir,
                 save_freq, resume, with_cuda, verbosity,
                 train_log_step, val_log_step, verbosity_iter,
                 valid_data_loader=None):
        super(Trainer,
              self).__init__(model, loss, metrics, optimizer, epochs,
                             training_name, save_dir, save_freq, with_cuda,
                             resume, verbosity, train_log_step, verbosity_iter)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader else False
        self.val_log_step = val_log_step

    def _train_epoch(self, epoch):
        """ Train an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        num_elements = len(self.data_loader)
        total_loss = 0
        batch_idx = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # start from the right point when restoring models
            if batch_idx < self.start_iteration:
                continue

            _log_iter_number = epoch * num_elements + batch_idx
            data, target = torch.FloatTensor(data), torch.LongTensor(target)
            data, target = Variable(data), Variable(target)
            if self.with_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            if (batch_idx % self.verbosity_iter == 0) & (self.verbosity == 2):
                self._logger.info('Epoch {:d} iteration {:d}'.format(
                    epoch, batch_idx))

            if (batch_idx % self.train_log_step) == 0:
                val = loss.item()
                self.model_logger.train.add_scalar('loss/iterations', val,
                                                   _log_iter_number)

                for i, metric in enumerate(self.metrics):
                    y_output = output.data.cpu().numpy()
                    y_output = np.argmax(y_output, axis=1)
                    y_target = target.data.cpu().numpy()
                    res_metric = metric(y_output, y_target)
                    self.model_logger.train.add_scalar(
                        'metrics/metric_{0}'.format(i), res_metric,
                        _log_iter_number)

            if (batch_idx % self.save_freq) == 0:
                if total_loss:
                    self._save_checkpoint(epoch, _log_iter_number,
                                          total_loss / batch_idx)

            if self.valid and (batch_idx % self.val_log_step == 0):
                # TODO: check what is returned form metrics
                val_loss, val_metrics = self._valid_epoch()

                self.model_logger.val.add_scalar('loss/iterations', val_loss,
                                                 _log_iter_number)

                for i, res_metric in enumerate(val_metrics):
                    self.model_logger.val.add_scalar(
                        'metrics/metric_{0}'.format(i), res_metric,
                        _log_iter_number)

                self.model.train()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)
        return avg_loss, batch_idx

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: loss and metrics
        """
        self.model.eval()
        if self.with_cuda:
            self.model.cuda()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for (data, target) in self.valid_data_loader:
            data, target = torch.FloatTensor(data), torch.LongTensor(target)
            data, target = Variable(data), Variable(target)
            if self.with_cuda:
                data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss = self.loss(output, target)
            total_val_loss += loss.item()

            for i, metric in enumerate(self.metrics):
                y_output = output.data.cpu().numpy()
                y_output = np.argmax(y_output, axis=1)
                y_target = target.data.cpu().numpy()
                total_val_metrics[i] += metric(y_output, y_target)

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        avg_val_metrics = (
            total_val_metrics / len(self.valid_data_loader)).tolist()
        return avg_val_loss, avg_val_metrics
