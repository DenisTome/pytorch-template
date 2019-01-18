# -*- coding: utf-8 -*-
"""
Created on Jun 12 06:38 2018

@author: Denis Tome'

"""
import torch
import numpy as np
from base.template import FrameworkClass

__all__ = [
    'ToTensor',
    'Convert'
]


class ToTensor(FrameworkClass):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        return {'img': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()}


class Convert(FrameworkClass):
    """Convert both hm and poses to Tensors."""

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = np.transpose(img, [2, 0, 1])
        label = np.transpose(label, [2, 0, 1])

        return {'img': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()}
