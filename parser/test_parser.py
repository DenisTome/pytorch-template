# -*- coding: utf-8 -*-
"""
General argument parser class
pre-defining most of the commong arguments.

@author: Denis Tome'

"""
from base.base_parser import BaseParser
import utils


class TestParser(BaseParser):
    """Train parser"""

    def __init__(self, description):
        super().__init__(description)

        # add default values ususally used for training
        # and can be individually changed as arguments
        self._add_batch_size(64)
        self._add_name('testing_model')
        self._add_resume(True)
        self._add_input_path()
        self._add_output_dir(utils.OUT_DIR)
        self._add_data_threads(8)
        self._add_cuda()
