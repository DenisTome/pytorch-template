import argparse
import logging
import torch.optim as optim
from model.model import Model
from model.loss import my_loss
from model.metric import my_metric, my_metric2
from data_loader.data_loader import DataLoader
from utils.util import split_validation
from trainer.trainer import Trainer

_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e',
    '--epochs',
    default=32,
    type=int,
    help='number of total epochs (default: 32)')
parser.add_argument(
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
    default=1000,
    type=int,
    help='after how many iterations to show info')
parser.add_argument(
    '--training_name',
    default='train_one',
    type=str,
    help='name of the training (default: train_one)')
parser.add_argument(
    '--save-dir',
    default='data/checkpoints',
    type=str,
    help='directory of saved model (default: model/saved)')
parser.add_argument(
    '--save-freq',
    default=10000,
    type=int,
    help='training checkpoint frequency in number of iterations(default: 1000)'
)
parser.add_argument(
    '--train_log_step',
    default=200,
    type=int,
    help='log frequency in number of iterations for training (default: 200)')
parser.add_argument(
    '--val_log_step',
    default=2000,
    type=int,
    help='log frequency in number of iterations for validation (default: 2000)'
)
parser.add_argument(
    '--data-dir',
    default='datasets',
    type=str,
    help='directory of training/testing data (default: datasets)')
parser.add_argument(
    '--validation-split',
    default=0.1,
    type=float,
    help='ratio of split validation data, [0.0, 1.0) (default: 0.0)')
parser.add_argument(
    '--no-cuda',
    action="store_true",
    help='use CPU in case there\'s no GPU support')


def main(args):
    # Model
    model = Model()
    model.summary()

    # Specifying loss function, metric(s), and optimizer
    loss = my_loss
    metrics = [my_metric, my_metric2]
    # define learning rate and other parameters depending on the optimizer
    optimizer = optim.Adam(model.parameters())

    # Data loader and validation split
    data_loader = DataLoader(args.data_dir, args.batch_size)
    data_loader, valid_data_loader = split_validation(data_loader,
                                                      args.validation_split)

    # Trainer instance
    trainer = Trainer(
        model,
        loss,
        metrics,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        training_name=args.training_name,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        resume=args.resume,
        verbosity=args.verbosity,
        verbosity_iter=args.verbosity_iter,
        train_log_step=args.train_log_step,
        val_log_step=args.val_log_step,
        with_cuda=not args.no_cuda)

    # Start training!
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
