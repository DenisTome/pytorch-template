# -*- coding: utf-8 -*-
"""
General argument parser class
pre-defining most of the commong arguments.

These parameters are automatically set by the
utils/config files and they can be then edited
as arguments.

@author: Denis Tome'

"""
from base import BaseParser


class TrainParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments

        # ------------------------------------------------------
        # --------------------- NN Related ---------------------

        # ------------------- Hyper-params -------------------
        self.add_learning_rate()
        self.add_batch_size()
        self.add_epochs()

        # ------------------- Generic -------------------
        self.add_checkpoint_freq()
        self.add_resume()
        self.add_reset()
        self.add_no_cuda()
        self.add_desc()

        # ------------------------------------------------------
        # --------------------- IO Related ---------------------
        self.add_dataset_input_type()
        self.add_name()
        self.add_train_dir()
        self.add_val_dir()
        self.add_datasets()
        self.add_verbose()
        self.add_standard_dirs()
        self.add_data_threads()

