# -*- coding: utf-8 -*-
"""
Test code where everything is specified by the user or
automatically assigned to the default value

@author: Denis Tome'

"""
import argparse
import logging
from torch.utils.data import DataLoader
from model.model import Model
from dataset_def import Dataset, ToTensor
from tester.tester import Tester
from utils import OUT_DIR

_LOGGER = logging.getLogger('main')

PARSER = argparse.ArgumentParser(description='Tester')
PARSER.add_argument(
    '-b',
    '--batch-size',
    default=64,
    type=int,
    help='mini-batch size (default: 512)')
PARSER.add_argument(
    '-n',
    '--name',
    default='train_HMAE',
    type=str,
    help='name of the training (default: train_AE)')
PARSER.add_argument(
    '-r',
    '--resume',
    default='',
    type=str,
    help='path to latest checkpoint (default: none)')
PARSER.add_argument(
    '--verbosity',
    default=2,
    type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
PARSER.add_argument(
    '--verbosity_iter',
    default=250,
    type=int,
    help='after how many iterations to show info')
PARSER.add_argument(
    '--save-dir',
    default=OUT_DIR,
    type=str,
    help='directory of saved model (default: model/checkpoints)')
PARSER.add_argument(
    '--test-path',
    required=True,
    type=str,
    help='file/dir with val-set annotations')
PARSER.add_argument(
    '--num-threads',
    default=8,
    type=int,
    help='number of threads for loading data (default: 8)')
PARSER.add_argument(
    '-s',
    '--sampling',
    default=5,
    type=int,
    help='sampling used on the test set (default: 5)')
PARSER.add_argument(
    '-d',
    '--description',
    default='',
    type=str,
    help='description of the training')
PARSER.add_argument(
    '--no-cuda',
    action="store_true",
    help='use CPU in case there\'s no GPU support')


def main(args):
    """Main"""

    # Model
    model = Model()
    model.summary()

    if not args.no_cuda:
        model.cuda()

    # Data loader
    test_set = Dataset(args.test_file,
                       transform=ToTensor())
    test_data_loader = DataLoader(test_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_threads)

    # Trainer instance
    tester = Tester(
        model,
        data_loader=test_data_loader,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        resume=args.resume,
        verbosity=args.verbosity,
        verbosity_iter=args.verbosity_iter,
        with_cuda=not args.no_cuda,
        output_name=args.name,
    )

    # Start training!
    tester.test()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(PARSER.parse_args())
