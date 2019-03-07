# -*- coding: utf-8 -*-
"""
Base model class

@author: Denis Tome'

"""
import torch.nn as nn
import numpy as np
from base.template import FrameworkClass


class BaseModel(nn.Module, FrameworkClass):
    """
    Base class for all model
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.name = self.__class__.__name__

    def forward(self, x):
        raise NotImplementedError

    def summary(self):
        """Summary of the model"""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        self._logger.info('Trainable parameters: %d', int(params))
