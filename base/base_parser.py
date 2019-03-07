# -*- coding: utf-8 -*-
"""
Base parser class for parsing arguments

@author: Denis Tome'

"""
import argparse
from base.template import FrameworkClass


class BaseParser(FrameworkClass):
    """Base parser class"""

    def __init__(self, description):
        """Initialization"""
        super().__init__()

        self.parser = argparse.ArgumentParser(description=description)

    def _add_batch_size(self, default):
        """Add batch-size argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-b',
            '--batch-size',
            default=default,
            type=int,
            help='mini-batch size (default: {:d})'.format(default))

    def _add_name(self, default):
        """Add name argument

        Arguments:
            default {str} -- default value
        """

        assert isinstance(default, str)

        self.parser.add_argument(
            '-n',
            '--name',
            default=default,
            type=str,
            help='output name (default: {})'.format(default))

    def _add_output_dir(self, default):
        """Add oytput directory argument

        Arguments:
            default {str} -- default value
        """

        assert isinstance(default, str)

        self.parser.add_argument(
            '-o',
            '--output',
            default=default,
            type=str,
            help='output directory path (default: {})'.format(default))

    def _add_input_path(self):
        """Add input argument"""

        self.parser.add_argument(
            '-i',
            '--input',
            required=True,
            type=str,
            help='input directory/file path')

    def _add_resume(self, required):
        """Add resume argument

        Arguments:
            required {bool} -- is this required
        """

        assert isinstance(required, bool)

        self.parser.add_argument(
            '-r',
            '--resume',
            default='init',
            required=required,
            type=str,
            help='input directory/file path containing the model')

    def _add_data_threads(self, default):
        """Add num threahds argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-t',
            '--num-threads',
            default=default,
            type=int,
            help='number of threads for dataset loader (default: {:d})'.format(
                default))

    def _add_cuda(self):
        """Add cuda argument"""

        self.parser.add_argument(
            '--no-cuda',
            action="store_true",
            help='use CPU in case there\'s no GPU support')

    def get_arguments(self):
        """Get arguments"""

        return self.parser.parse_args()
