# -*- coding: utf-8 -*-
"""
Created on Jan 18 17:32 2019

@author: Denis Tome'

Example of Model definition class

"""
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Model(BaseModel):
    """Model"""

    def __init__(self):
        """Init"""
        super().__init__()

        self.cnn = None
        self.fc = None
        self._build_model()

    def _build_model(self):
        """Building model"""

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(16))
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        """Forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output
        """

        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)
