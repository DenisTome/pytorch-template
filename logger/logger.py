# -*- coding: utf-8 -*-
"""
Created on Jun 12 06:38 2018

@author: Denis Tome'

"""
import os
import torch
from torch.autograd import Variable
from utils.util import ensure_dir
from tensorboardX import SummaryWriter

__all__ = [
    'Logger',
    'SingleLogger'
]


class Logger(object):
    """
    Logger, used by BaseTrainer to save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name

        self.train = SummaryWriter(
            os.path.join(self.dir_path, 'train'), comment=self.training_name)
        self.val = SummaryWriter(
            os.path.join(self.dir_path, 'val'), comment=self.training_name)

    def add_graph_definition(self, model):
        dummy_input = Variable(torch.rand(1, 3, 224, 224))
        self.train.add_graph(model, dummy_input)

    def close_all(self):
        self.train.close()
        self.val.close()


class SingleLogger(object):
    """
    Logger, used by BaseTrainer to save training history
    """

    def __init__(self, dir_path, training_name):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name

        self.train = SummaryWriter(
            os.path.join(self.dir_path, self.training_name), comment=self.training_name)

    def close_all(self):
        self.train.close()
