# -*- coding: utf-8 -*-
"""
General argument parser class
pre-defining most of the commong arguments.

@author: Denis Tome'

"""
from base import BaseParser
import utils


class TrainParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_learning_rate(0.0001)
        self._add_batch_size(64)
        self._add_epochs(100)
        self._add_name('generic_training')
        self._add_lr_decay()
        self._add_resume(False, '-r', '--resume')
        self._add_input()
        self._add_validation(True)
        self._add_output_dir(utils.DIRS.checkpoint)
        self._add_model_checkpoints(1000)
        self._add_verbose(50, 1000, 200, 20)
        self._add_data_threads(8)
        self._add_cuda()
        self._add_reset()
