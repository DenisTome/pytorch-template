# -*- coding: utf-8 -*-
"""
Train model

@author: Denis Tome'

"""
import datetime
from base.base_trainer import BaseTrainer
from utils.draw import Drawer, Style

__all__ = [
    'Trainer'
]


class Trainer(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self, model, loss, regularizer, metrics, data_loader,
                 optimizer, reg_weights, epochs, training_name, img_log_step,
                 save_dir, save_freq, resume, with_cuda, verbosity,
                 train_log_step, val_log_step, verbosity_iter,
                 reset=False, eval_epoch=False, lr_decay=None,
                 valid_data_loader=None, description=None):
        super().__init__(model, loss, metrics, optimizer, epochs,
                         training_name, save_dir, save_freq, with_cuda, resume,
                         verbosity, train_log_step, verbosity_iter, reset, eval_epoch)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader else False
        self.val_log_step = val_log_step
        self.lr_decay = lr_decay
        self.img_log_step = img_log_step
        self.regularizer = regularizer
        self.reg_weights = reg_weights
        self.description = description
        self.len_trainset = len(self.data_loader)
        self.drawer = Drawer(Style.EQ_AXES)

    def _summary_info(self):
        """
        Summary file to differentiate between all the different trainings
        """
        info = dict()
        info['batch_size'] = self.batch_size
        info['creation'] = str(datetime.datetime.now())
        info['size_dataset'] = len(self.data_loader)
        info['size_valset'] = len(self.valid_data_loader)
        info['start_lr'] = utils.get_optimizer_lr(self.optimizer)
        info['lr_decay'] = True if self.lr_decay else False
        if self.lr_decay:
            info['lr_decay_rate'] = self.lr_decay.decay_rate
            info['lr_decay_step'] = self.lr_decay.decay_step
            info['lr_decay_mode'] = self.lr_decay.mode
        for rid, reg in enumerate(self.regularizer):
            info['reg_{}_weight'.format(reg.__name__)] = self.reg_weights[rid]
        if self.description:
            info['description'] = self.description
        return info

    def _train_epoch(self, epoch):
        """Train model for one epoch

        Arguments:
            epoch {int} -- epoch number

        Returns:
            float -- epoch error
        """

        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        batch_idx = self.start_iteration

        for (data, target) in self.data_loader:
            data = self._get_var(data)
            target = self._get_var(target)

            # set right learning rate depending on decay
            if self.lr_decay:
                self.lr_decay.update_lr(self.global_step)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            reg_values = []
            for rid, reg in enumerate(self.regularizer):
                reg_value = reg(data, target)
                loss.add(reg_value.mul(self.reg_weights[rid]))
                reg_values.append(reg_value)
            loss.backward()
            self.optimizer.step()

            if (batch_idx % self.verbosity_iter == 0) & (self.verbosity == 2):
                self._logger.info('Epoch {:d} \t batch_id {:d} / {:d} \t iteration {:d}'.format(
                    epoch, batch_idx, self.len_trainset, self.global_step))

            if (batch_idx % self.train_log_step) == 0:
                val = loss.item()
                self.model_logger.train.add_scalar('loss/iterations', val,
                                                   self.global_step)

                for rid, reg in enumerate(self.regularizer):
                    self.model_logger.train.add_scalar('regularizer/{}'.format(reg.__name__),
                                                       reg_values[rid].data.cpu(
                    ).item(),
                        self.global_step)

                for metric in self.metrics:
                    y_output = data.data.cpu().numpy()
                    y_target = target.data.cpu().numpy()
                    metric.log(pred=y_output,
                               gt=y_target,
                               logger=self.model_logger.train,
                               iteration=self.global_step)

            if (batch_idx % self.img_log_step) == 0 and self.img_log_step > -1:
                y_output = data.data.cpu().numpy()
                y_target = target.data.cpu().numpy()

                img_poses = self.drawer.poses_3d(y_output[0], y_target[0])
                self.model_logger.train.add_image('3d_poses',
                                                  img_poses,
                                                  self.global_step)

            if (batch_idx % self.save_freq) == 0:
                if total_loss:
                    self._save_checkpoint(epoch, self.global_step,
                                          total_loss / batch_idx)

            if self.valid and (batch_idx % self.val_log_step == 0) and (batch_idx != 0):
                self._logger.info('Evaluating performance of evaluation set')
                val_loss, val_metrics = self._valid_epoch()

                self.model_logger.val.add_scalar('loss/iterations', val_loss,
                                                 self.global_step)

                for i, metric in enumerate(self.metrics):
                    metric.log_res(logger=self.model_logger.val,
                                   iteration=self.global_step,
                                   error=val_metrics[i])

                self._update_summary(self.global_step,
                                     val_loss,
                                     val_metrics)
                self.model.train()

            batch_idx += 1
            self.global_step += 1
            total_loss += loss.item()

        avg_loss = total_loss / (batch_idx - self.start_iteration)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)

        return avg_loss

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: loss and metrics
        """
        self.model.eval()
        if self.with_cuda:
            self.model.cuda()

        total_val_loss = 0
        total_val_metrics = [None] * len(self.metrics)
        num_elems = len(self.valid_data_loader)
        batch_idx = 0

        for (data, target) in self.valid_data_loader:
            data = self._get_var(data)
            target = self._get_var(target)

            output = self.model(data)
            loss = self.loss(output, target)
            total_val_loss += loss.item()

            y_output = data.data.cpu().numpy()
            y_target = target.data.cpu().numpy()
            for i, metric in enumerate(self.metrics):
                total_val_loss[i] = metric.add_results(total_val_loss[i],
                                                       y_output,
                                                       y_target)

            batch_idx += 1

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        for i, metric in enumerate(self.metrics):
            total_val_metrics[i] /= len(self.valid_data_loader)
        return avg_val_loss, total_val_metrics
