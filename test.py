# -*- coding: utf-8 -*-
"""
Test code where everything is specified by the user or
automatically assigned to the default value

@author: Denis Tome'

"""
from parser import TestParser
from torch.utils.data import DataLoader
from logger.console_logger import ConsoleLogger
from model import Model
from dataset_def import Dataset, ToTensor
from tester.tester import Tester

LOGGER = ConsoleLogger("Main")
PARSER = TestParser('Tester')


def main(args):
    """Main"""

    LOGGER.info('Testing...')

    # Model
    model = Model()
    model.summary()

    if not args.no_cuda:
        model.cuda()

    # Data loader
    test_set = Dataset(args.input,
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
        save_dir=args.output,
        output_name=args.name,
        resume=args.resume,
        verbosity=args.verbosity,
        verbosity_iter=args.verbosity_iter,
        with_cuda=not args.no_cuda,
    )

    # Start training!
    tester.test()

    LOGGER.info('Done.')


if __name__ == '__main__':
    main(PARSER.get_arguments())
