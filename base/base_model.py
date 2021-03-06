# -*- coding: utf-8 -*-
"""
Base model class

@author: Denis Tome'

"""

__version__ = "0.1.2"

import torch.nn as nn
import numpy as np
from logger.console_logger import ConsoleLogger


class BaseModel(nn.Module):
    """
    Base class for all model
    """

    def __init__(self):
        """Init"""
        super().__init__()
        self.model = None

        self.name = self.__class__.__name__
        self._logger = ConsoleLogger(self.__class__.__name__)

    def summary(self) -> None:
        """Summary of the model"""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        self._logger.info('Trainable parameters: %d', int(params))
