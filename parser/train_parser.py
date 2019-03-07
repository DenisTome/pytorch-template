# -*- coding: utf-8 -*-
"""
General argument parser class
pre-defining most of the commong arguments.

@author: Denis Tome'

"""
from base.base_parser import BaseParser
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
        self._add_resume(False)
        self._add_input_path()
        self._add_validation(True)
        self._add_output_dir(utils.CHKP_DIR)
        self._add_model_checkpoints(1000)
        self._add_verbose(50, 1000, 200)
        self._add_data_threads(8)
        self._add_cuda()
        self._add_reset()

    def _add_learning_rate(self, default):
        """Add learning rate argument

        Arguments:
            default {float} -- default value
        """

        assert isinstance(default, float)

        self.parser.add_argument(
            '-lr',
            '--learning-rate',
            default=default,
            type=float,
            help='learning rate (default: {:.6f})'.format(default))

    def _add_epochs(self, default):
        """Add epochs argument

        Arguments:
            default {int} -- default value
        """

        assert isinstance(default, int)

        self.parser.add_argument(
            '-e',
            '--epochs',
            default=default,
            type=int,
            help='number of epochs (default: {:d})'.format(default))

    def _add_validation(self, required):
        """Add validation path argument

        Arguments:
            required {boold} -- is required
        """

        self.parser.add_argument(
            '--validation',
            required=required,
            default='',
            type=str,
            help='Path to validation data')

    def _add_reset(self):
        """Add reset argument"""

        self.parser.add_argument(
            '--reset',
            action="store_true",
            help='reset global information about restored model')

    def _add_lr_decay(self):
        """Add learning rate related"""

        self.parser.add_argument(
            '--no-lr-decay',
            action="store_true",
            help='don\'t use learning rate decay')
        self.parser.add_argument(
            '--lr_decay_rate',
            default=0.95,
            type=float,
            help='learning rate decay rate (default = 0.95)')
        self.parser.add_argument(
            '--lr_decay_step',
            default=3000,
            type=float,
            help='learning rate decay step (default = 3000)')

    def _add_model_checkpoints(self, iterations):
        """Add argument to save model every n iterations

        Arguments:
            iterations {int} -- number of iterations
        """

        self.parser.add_argument(
            '--save-freq',
            default=iterations,
            type=int,
            help='training checkpoint frequency in iterations(default: {:d})'.format(iterations))

    def _add_verbose(self, train, val, image):
        """Add arguments for verbose

        Arguments:
            train {int} -- number of iterations
            val {int} -- number of iterations
            image {int} -- number of iterations
        """

        self.parser.add_argument(
            '-v',
            '--verbosity',
            default=2,
            type=int,
            help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
        self.parser.add_argument(
            '--train-log-step',
            default=train,
            type=int,
            help='log frequency for training (default: {:d})'.format(train))
        self.parser.add_argument(
            '--val-log-step',
            default=val,
            type=int,
            help='log frequency for validation (default: {:d})'.format(val))
        self.parser.add_argument(
            '--img-log-step',
            default=image,
            type=int,
            help='log frequency for images (default: {:d})'.format(image))
        self.parser.add_argument(
            '--eval-epoch',
            action="store_true",
            help='show model evaluation at the end of each epoch')
