# -*- coding: utf-8 -*-
"""
Train model

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
from tqdm import tqdm
from base import BaseTrainer
from utils.draw import PLTPoseVisualizer

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

        self._drawer = PLTPoseVisualizer()

    def _train_epoch(self, epoch: int) -> float:
        """Train model for one epoch

        Args:
            epoch (int): epoch number

        Returns:
            float: epoch error
        """

        self._model.train()

        total_loss = 0
        pbar = tqdm(self._train_loader)
        for bid, (data, target) in enumerate(pbar):

            # load data in GPU
            data = self._get_var(data)
            target = self._get_var(target)

            # ------------------------------------------------------------------
            # -------------------- Forward + Backward passes  ------------------
            # ------------------------------------------------------------------

            self._opt.zero_grad()

            output = self._model(data)
            loss = self._loss(output, target)
            loss.backward()
            self._opt.step()

            # -----------------------------------------------------------
            # ------------------------ Train log ------------------------
            # -----------------------------------------------------------

            if (self._global_step % self._train_log_step) == 0:

                val = loss.item()
                self._model_logger.train.add_scalar('loss/iterations', val,
                                                    self._global_step)

                for metric in self._metrics:
                    y_output = data.data.cpu().numpy()
                    y_target = target.data.cpu().numpy()
                    metric.log(pred=y_output,
                               gt=y_target,
                               logger=self._model_logger.train,
                               iteration=self._global_step)

            # ------------------------------------------------------
            # ----------------------- Images -----------------------

            if (self._global_step % self._img_log_step) == 0 and self._img_log_step > -1:
                y_output = data.data.cpu().numpy()
                y_target = target.data.cpu().numpy()

                img_poses = self._drawer.plot3DPose(y_output[0], y_target[0])
                self._model_logger.train.add_image('3d_poses',
                                                   img_poses.transpose(
                                                       [2, 0, 1]),
                                                   self._global_step)

            # -------------------------------------------------------
            # --------------------- Checkpoints ---------------------
            # -------------------------------------------------------

            if (self._global_step % self._save_freq) == 0:
                if total_loss:
                    self._save_checkpoint(epoch, self._global_step,
                                          total_loss / bid)

            # ------------------------------------------------------
            # --------------------- Validation ---------------------
            # ------------------------------------------------------

            if self._val_loader:
                if (self._global_step % self._val_log_step == 0) and (self._global_step > 0):
                    self._logger.info(
                        'Evaluating performance of evaluation set')
                    val_loss = self._valid_epoch()

                    self._model_logger.val.add_scalar('loss/iterations', val_loss,
                                                      self._global_step)

                    self._model.train()

            self._global_step += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(self._train_loader)
        self._model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)

        return avg_loss

    def _valid_epoch(self):
        """Validate model

        Returns:
            loss -- validation loss
        """

        self._model.eval()
        total_val_loss = 0

        pbar = tqdm(self._val_loader)
        pbar.set_description(' Validation')
        for (data, target) in pbar:

            # Load data in GPU
            data = self._get_var(data)
            target = self._get_var(target)

            output = self._model(data)
            loss = self._loss(output, target)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self._val_loader)

        return avg_val_loss
