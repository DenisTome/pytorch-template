# -*- coding: utf-8 -*-
"""
Created on Jul 03 16:21 2018

@author: Denis Tome'

Test code where everything is specified by the user or
automatically assigned to the default value.

"""
import argparse
import logging
from model.model import Model
from dataset_def import Dataset, ToTensor
from tester.tester import Tester
from utils import OUT_DIR
from torch.utils.data import DataLoader

_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='Tester')
parser.add_argument(
    '-b',
    '--batch-size',
    default=64,
    type=int,
    help='mini-batch size (default: 512)')
parser.add_argument(
    '-n',
    '--name',
    default='train_HMAE',
    type=str,
    help='name of the training (default: train_AE)')
parser.add_argument(
    '-r',
    '--resume',
    default='',
    type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2,
    type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--verbosity_iter',
    default=250,
    type=int,
    help='after how many iterations to show info')
parser.add_argument(
    '--save-dir',
    default=OUT_DIR,
    type=str,
    help='directory of saved model (default: model/checkpoints)')
parser.add_argument(
    '--test-path',
    required=True,
    type=str,
    help='file/dir with val-set annotations')
parser.add_argument(
    '--num-threads',
    default=8,
    type=int,
    help='number of threads for loading data (default: 8)')
parser.add_argument(
    '-s',
    '--sampling',
    default=5,
    type=int,
    help='sampling used on the test set (default: 5)')
parser.add_argument(
    '-d',
    '--description',
    default='',
    type=str,
    help='description of the training')
parser.add_argument(
    '--no-cuda',
    action="store_true",
    help='use CPU in case there\'s no GPU support')


def main(args):
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
    main(parser.parse_args())
