# -*- coding: utf-8 -*-
"""
Train model

@author: Denis Tome'

"""
import datetime
from tqdm import tqdm
from base import BaseTrainer
from model.modules import LRDecay
from utils.draw import PLTPoseVisualizer
import utils

__all__ = [
    'Trainer'
]


class Trainer(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self, *args, **kwargs):
        """Init"""

        super().__init__(*args, **kwargs)

        self.drawer = PLTPoseVisualizer()
        # add class specific stuff

    def _train_epoch(self, epoch):
        """Train model for one epoch

        Arguments:
            epoch {int} -- epoch number

        Returns:
            float -- epoch error
        """

        self.model.train()

        total_loss = 0
        pbar = tqdm(self.train_loader)
        for bid, (data, target) in enumerate(pbar):

            # load data in GPU
            data = self._get_var(data)
            target = self._get_var(target)

            # ------------------------------------------------------------------
            # -------------------- Forward + Backward passes  ------------------
            # ------------------------------------------------------------------

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # -----------------------------------------------------------
            # ------------------------ Train log ------------------------
            # -----------------------------------------------------------

            if (bid % self.train_log_step) == 0:

                val = loss.item()
                self.model_logger.train.add_scalar('loss/iterations', val,
                                                   self.global_step)

                for metric in self.metrics:
                    y_output = data.data.cpu().numpy()
                    y_target = target.data.cpu().numpy()
                    metric.log(pred=y_output,
                               gt=y_target,
                               logger=self.model_logger.train,
                               iteration=self.global_step)

            # ------------------------------------------------------
            # ----------------------- Images -----------------------

            if (bid % self.img_log_step) == 0 and self.img_log_step > -1:
                y_output = data.data.cpu().numpy()
                y_target = target.data.cpu().numpy()

                img_poses = self.drawer.poses_3d(y_output[0], y_target[0])
                self.model_logger.train.add_image('3d_poses',
                                                  img_poses.transpose([2, 0, 1]),
                                                  self.global_step)

            # -------------------------------------------------------
            # --------------------- Checkpoints ---------------------
            # -------------------------------------------------------

            if (self.global_step % self.save_freq) == 0:
                if total_loss:
                    self._save_checkpoint(epoch, self.global_step,
                                          total_loss / bid)

            # ------------------------------------------------------
            # --------------------- Validation ---------------------
            # ------------------------------------------------------

            if self.val_loader and (bid % self.val_log_step == 0) and (bid > 0):
                self._logger.info('Evaluating performance of evaluation set')
                val_loss = self._valid_epoch()

                self.model_logger.val.add_scalar('loss/iterations', val_loss,
                                                 self.global_step)

                self.model.train()

            self.global_step += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)

        return avg_loss

    def _valid_epoch(self):
        """Validate model

        Returns:
            loss -- validation loss
        """

        self.model.eval()
        if self.with_cuda:
            self.model.cuda()

        total_val_loss = 0
        
        pbar = tqdm(self.val_loader)
        pbar.set_description(' Validation')
        for (data, target) in pbar:

            # Load data in GPU
            data = self._get_var(data)
            target = self._get_var(target)

            output = self.model(data)
            loss = self.loss(output, target)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        
        return avg_val_loss
