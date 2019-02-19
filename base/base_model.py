# -*- coding: utf-8 -*-
"""
Base model class

@author: Denis Tome'

"""
import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all model
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__

    def forward(self, x):
        raise NotImplementedError

    def summary(self):
        """Summary of the model"""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        self._logger.info('Trainable parameters:{}'.format(params))
