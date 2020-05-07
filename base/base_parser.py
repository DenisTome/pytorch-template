# -*- coding: utf-8 -*-
"""
Base parser class for parsing arguments.
It automatically fatches the default values from the data/parameters.yml
file and are overwritten from the arguments provided by the user.

@author: Denis Tome'

"""
import argparse
from base.template import FrameworkClass
from utils import config, model_config, machine


class BaseParser(FrameworkClass):
    """Base parser class"""

    def __init__(self, description):
        """Initialization"""
        super().__init__()

        self.parser = argparse.ArgumentParser(
            description=description)
        self.param = model_config

    def add_learning_rate(self):
        """Add learning rate argument"""

        val = float(self.param.nn.learning_rate)
        self.parser.add_argument(
            '-lr',
            '--learning-rate',
            default=val,
            type=float,
            help='learning rate (default: {:.6f})'.format(val))

    def add_weight_decay(self):
        """Add weight decay argument"""

        val = float(self.param.opt.weight_decay)
        self.parser.add_argument(
            '--weight-decay',
            default=val,
            type=float,
            help='weight-decay value (default: {:.6f})'.format(val))

    def add_batch_size(self):
        """Add batch-size argument"""

        val = int(self.param.nn.batch_size)
        self.parser.add_argument(
            '-b',
            '--batch-size',
            default=val,
            type=int,
            help='mini-batch size (default: {:d})'.format(val))

    def add_epochs(self):
        """Add epochs argument"""

        val = int(self.param.nn.epochs)
        self.parser.add_argument(
            '-e',
            '--epochs',
            default=val,
            type=int,
            help='number of epochs (default: {:d})'.format(val))

    def add_checkpoint_freq(self):
        """Add argument to save model every n iterations"""

        val = int(self.param.nn.save_freq)
        self.parser.add_argument(
            '--save-freq',
            default=val,
            type=int,
            help='training checkpoint frequency in iterations(default: {:d})'.format(val))

    def add_resume(self):
        """Add resume argument"""

        val = self.param.nn.resume
        self.parser.add_argument(
            '-r',
            '--resume',
            default=val,
            type=str,
            help='input directory/file path containing the model')

    def add_reset(self):
        """Add reset argument"""

        self.parser.add_argument(
            '--reset',
            action="store_true",
            help='reset global information about restored model')

    def add_no_cuda(self):
        """Add cuda argument"""

        self.parser.add_argument(
            '--no-cuda',
            action="store_true",
            help='use CPU in case there\'s no GPU support')

    def add_desc(self):
        """Add desc to name"""

        self.parser.add_argument(
            '--desc',
            action="store_true",
            help='add description to name')

        self.parser.add_argument(
            '--desc-str',
            default='',
            type=str,
            help='descriptor appendinx (default: "")')

    def add_mode(self):
        """Add prediction mode.
        This can be either dataset to same dataset
        or dataset_a to dataset_b
        """

        val = self.param.nn.mode
        self.parser.add_argument(
            '-m',
            '--model-mode',
            default=val,
            type=str,
            help='prediction mode (default: {})'.format(val))

    # ------------------------------------------------------
    # --------------------- IO Related ---------------------
    # ------------------------------------------------------

    def add_dataset_input_type(self):
        """Add dataset input type argument"""

        val = machine.datasets.type
        self.parser.add_argument(
            '--dataset-input-type',
            default=val,
            type=str,
            help='dataset input type (default: {})'.format(val))

    def add_name(self):
        """Add name argument"""

        val = self.param.opt.name
        self.parser.add_argument(
            '-n',
            '--name',
            default=val,
            type=str,
            help='output name (default: {})'.format(val))

    @staticmethod
    def _replace_placeholder(path):
        """Filter path if there are placeholders"""
        if '${dataset_type}' in path:
            path = path.replace('${dataset_type}', machine.datasets.type)

        return path

    def add_train_dir(self):
        """Add train dir"""

        val = self._replace_placeholder(machine.dirs.train_dir)
        self.parser.add_argument(
            '--train-dir',
            default=val,
            type=str,
            help='input directory/file path (default: {})'.format(val))

    def add_test_dir(self):
        """Add train dir"""

        val = self._replace_placeholder(machine.dirs.test_dir)
        self.parser.add_argument(
            '--test-dir',
            default=val,
            type=str,
            help='input directory/file path (default: {})'.format(val))

    def add_val_dir(self):
        """Add train dir"""

        val = self._replace_placeholder(machine.dirs.val_dir)
        self.parser.add_argument(
            '--val-dir',
            default=val,
            type=str,
            help='input directory/file path (default: {})'.format(val))

    def add_datasets(self):
        """Add datasets to use"""

        val = self.param.opt.datasets
        self.parser.add_argument(
            '--datasets',
            default=val,
            type=str,
            help='dataset aliases to be used separated by commas (default: {})'.format(val))

    def add_dataset(self):
        """Add datasets to use"""

        val = self.param.opt.dataset
        self.parser.add_argument(
            '--dataset',
            default=val,
            type=str,
            help='dataset alias to be used (default: {})'.format(val))

    def add_verbose(self):
        """Add arguments for verbose"""

        val = int(self.param.opt.train_log_step)
        self.parser.add_argument(
            '--train-log-step',
            default=val,
            type=int,
            help='how frequently show loss during training (default: {:d})'.format(val))

        val = int(self.param.opt.val_log_step)
        self.parser.add_argument(
            '--val-log-step',
            default=val,
            type=int,
            help='how frequently compute loss on val set (default: {:d})'.format(val))

        val = int(self.param.opt.img_log_step)
        self.parser.add_argument(
            '--img-log-step',
            default=val,
            type=int,
            help='how frequently save pose images (default: {:d})'.format(val))

        self.parser.add_argument(
            '--eval-epoch',
            action="store_true",
            help='show model evaluation at the end of each epoch')

    def add_standard_dirs(self):
        """Add output and checkpoint directories argument"""

        self.parser.add_argument(
            '-o',
            '--output',
            default=config.dirs.output,
            type=str,
            help='output directory path (default: {})'.format(config.dirs.output))

        self.parser.add_argument(
            '-s',
            '--checkpoint-dir',
            default=config.dirs.checkpoint,
            type=str,
            help='checkpoint directory path (default: {})'.format(config.dirs.checkpoint))

    def add_data_threads(self):
        """Add num threahds argument"""

        val = int(self.param.opt.num_workers)
        self.parser.add_argument(
            '-t',
            '--num-workers',
            default=val,
            type=int,
            help='number of workers for dataset loader (default: {:d})'.format(val))

    # ------------------------------------------------------
    # ------------------- Get Arguments --------------------
    # ------------------------------------------------------

    def get_arguments(self):
        """Get arguments"""

        return self.parser.parse_args()
