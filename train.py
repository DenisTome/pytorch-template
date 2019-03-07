# -*- coding: utf-8 -*-
"""
Train code where everything is specified by the user or
automatically assigned to the default value

@author: Denis Tome'

"""
from parser.train_parser import TrainParser
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.model import Model
from model.modules.loss import ae_loss as loss
from model.modules.metric import AvgPoseError
from model.modules.lr_decay import LRDecay
from logger.console_logger import ConsoleLogger
from dataset_def import Dataset
from dataset_def import Convert, ToTensor
from trainer.trainer import Trainer

LOGGER = ConsoleLogger('Main')
PARSER = TrainParser('Trainer')


def main(args):
    """Main"""

    LOGGER.info('Staring training...')

    # Model
    model = Model()
    model.summary()

    # Specifying loss function, metric(s), and optimizer
    metrics = [AvgPoseError()]
    regularizers = None
    reg_weights = 0

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
    train_set = Dataset(args.input,
                        transform=data_transform)
    train_data_loader = DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    # Val-set
    val_set = Dataset(args.validation,
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
        training_name=args.name,
        save_dir=args.output,
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
    )

    # Start training!
    trainer.train()

    LOGGER.info('Done.')


if __name__ == '__main__':
    main(PARSER.get_arguments())
