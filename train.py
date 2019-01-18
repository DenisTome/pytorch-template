# -*- coding: utf-8 -*-
"""
Created on Jul 03 16:21 2018

@author: Denis Tome'

Train code where everything is specified by the user or
automatically assigned to the default value.

"""
import os
import argparse
import logging
import torch.optim as optim
from model.model import Model
from model.modules.loss import ae_loss as loss
from model.modules.metric import AvgPoseError
from model.modules.regularizer import limb_length
from model.modules.lr_decay import LRDecay
from dataset_def import Dataset
from dataset_def import Convert, ToTensor
from trainer.trainer import Trainer
from utils import CHKP_DIR
from torchvision import transforms
from torch.utils.data import DataLoader

_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument(
    '-lr',
    '--learning-rate',
    default=1e-4,
    type=float,
    help='learning rate')
parser.add_argument(
    '-b',
    '--batch-size',
    default=64,
    type=int,
    help='mini-batch size (default: 64)')
parser.add_argument(
    '-e',
    '--epochs',
    default=100,
    type=int,
    help='number of total epochs (default: 100)')
parser.add_argument(
    '-n',
    '--training_name',
    default='train',
    type=str,
    help='name of the training (default: train_AE)')
parser.add_argument(
    '--no-lr-decay',
    action="store_true",
    help='don\'t use learning rate decay')
parser.add_argument(
    '--lr_decay_rate',
    default=0.95,
    type=float,
    help='learning rate decay rate (default = 0.95)')
parser.add_argument(
    '--lr_decay_step',
    default=3000,
    type=float,
    help='learning rate decay step (default = 3000)')
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
    default=CHKP_DIR,
    type=str,
    help='directory of saved model (default: model/checkpoints)')
parser.add_argument(
    '--save-freq',
    default=10000,
    type=int,
    help='training checkpoint frequency in number of iterations(default: 10000)')
parser.add_argument(
    '--img-log-step',
    default=250,
    type=int,
    help='number of iterations for log images (default: 1000)')
parser.add_argument(
    '--train-log-step',
    default=100,
    type=int,
    help='log frequency in number of iterations for training (default: 500)')
parser.add_argument(
    '--val-log-step',
    default=250,
    type=int,
    help='log frequency in number of iterations for validation (default: 2000)')
parser.add_argument(
    '--train-path',
    required=True,
    type=str,
    help='file/directory of training data')
parser.add_argument(
    '--val-path',
    required=True,
    type=str,
    help='file/directory of validation data')
parser.add_argument(
    '--eval-epoch',
    action="store_true",
    help='flag to apply input noise during training')
parser.add_argument(
    '--num-threads',
    default=8,
    type=int,
    help='number of threads for loading data (default: 8)')
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
parser.add_argument(
    '--reset',
    action="store_true",
    help='reset global information about previous model')


def main(args):
    # Model
    model = Model()
    model.summary()

    # Specifying loss function, metric(s), and optimizer
    metrics = [AvgPoseError()]
    regularizers = [limb_length]
    reg_weights = [args.lreg_weight]

    if not args.no_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)

    lr_decay = None
    if not args.no_lr_decay:
        lr_decay = LRDecay(optimizer,
                           lr=args.learning_rate,
                           decay_rate=args.lr_decay_rate,
                           decay_steps=args.lr_decay_step)

    data_transform = transforms.Compose([
        Convert(),
        ToTensor()
    ])

    # Train-set
    train_set = Dataset(args.train_path,
                        transform=data_transform)
    train_data_loader = DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    # Val-set
    val_set = Dataset(args.val_path,
                      transform=data_transform)
    val_data_loader = DataLoader(val_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_threads)

    # Trainer instance
    trainer = Trainer(
        model,
        loss,
        regularizers,
        metrics,
        data_loader=train_data_loader,
        valid_data_loader=val_data_loader,
        optimizer=optimizer,
        reg_weights=reg_weights,
        lr_decay=lr_decay,
        epochs=args.epochs,
        training_name=args.training_name,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        resume=args.resume,
        verbosity=args.verbosity,
        verbosity_iter=args.verbosity_iter,
        train_log_step=args.train_log_step,
        img_log_step=args.img_log_step,
        val_log_step=args.val_log_step,
        with_cuda=not args.no_cuda,
        reset=args.reset,
        eval_epoch=args.eval_epoch,
        description=args.description
    )

    # Start training!
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
