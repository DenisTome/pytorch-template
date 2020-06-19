# -*- coding: utf-8 -*-
"""
Train code where everything is specified by the user or
automatically assigned to the default value

@author: Denis Tome'

"""
from parser import TrainParser
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from model import Model
from model.modules.optimizer import Optimizer
from model.modules import PoseError
from logger.console_logger import ConsoleLogger
from dataset_def import Dataset, SetType
from dataset_def import DatasetInputFormat, OutputData
from dataset_def import transformations as trsf
from trainer import Trainer
from utils import config

LOGGER = ConsoleLogger('Main')
PARSER = TrainParser('Trainer')


def get_input_type(input_type: str) -> DatasetInputFormat:
    """Get dataset input type as DatasetInputFormat

    Arguments:
        input_type {str} -- string of input type

    Raises:
        RuntimeError: input type not found

    Returns:
        DatasetInputFormat -- input type
    """

    for t in DatasetInputFormat:
        if t.value == input_type:
            return t

    raise RuntimeError(
        'Dataset input type {} not supported'.format(input_type))


def get_data_loaders(args):
    """Create data loaders

    Arguments:
        args {dict} -- arguments

    Returns:
        DataLoader -- train data loader
        DataLoader -- val data loader
    """

    d_names = args.datasets.split(',')
    data_transform = {}
    for d_name in d_names:
        assert d_name in config.dataset.supported

        data_trsf = trsf.Compose(
            [trsf.Translation(d_name),
             trsf.Rotation(d_name),
             trsf.QuaternionToR(d_name),
             trsf.Align(d_name)]
        )

        data_transform.update({d_name: data_trsf})

    input_type = get_input_type(args.dataset_input_type)

    # ------------------- Data selection -------------------
    selection = OutputData.P3D | OutputData.DID | OutputData.ROT

    # ------------------- Train-set -------------------
    multi_datasets = []
    for d_name in d_names:
        multi_datasets.append(os.path.join(args.train_dir, d_name))

    train_set = Dataset(multi_datasets,
                        input_type=input_type,
                        transf=data_transform,
                        out_data=selection)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    # ------------------- Val-set -------------------

    multi_datasets = []
    for d_name in d_names:
        val_path = os.path.join(args.val_dir, d_name)
        if os.path.exists(val_path):
            multi_datasets.append(val_path)

    val_set = Dataset(multi_datasets,
                      input_type=input_type,
                      transf=data_transform,
                      out_data=selection,
                      set_type=SetType.VAL)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)

    return train_loader, val_loader


def main(args):
    """Main"""

    LOGGER.info('Staring training...')

    # Model
    model = Model()
    model.summary()

    # Specifying loss function, metric(s), and optimizer
    metrics = [PoseError()]

    if not args.no_cuda:
        model.cuda()

    # ------------------- Optimization -------------------

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)

    # ------------------- Data loader -------------------

    train_loader, val_loader = get_data_loaders(args)

    # ------------------- Trainer -------------------

    # Trainer instance
    trainer = Trainer(
        model=model,
        loss=MSELoss(),
        metrics=metrics,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        **vars(args)
    )

    # Start training!
    trainer.train()

    LOGGER.info('Done.')


if __name__ == '__main__':
    main(PARSER.get_arguments())
