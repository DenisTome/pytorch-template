# -*- coding: utf-8 -*-
"""
Transformations

@author: Denis Tome'

"""
import torch
from torchvision import transforms
import numpy as np
from base import BaseTransformation
from base.base_dataset import OutputData

__all__ = [
    'ToTensor',
    'ImageNormalization'
]


class ToTensor(BaseTransformation):
    """Turn array to tensor"""

    _SCOPE = OutputData.ALL

    def to_tensor(self, data):
        """Return tensor from numpy input"""

        if data is None:
            return data

        for key, elem in data.items():
            if isinstance(elem, torch.Tensor):
                continue

            assert isinstance(elem, np.ndarray, np.generic)
            data[key] = torch.Tensor(elem)

        return data


class ImageNormalization(BaseTransformation):
    """Transformation class to normalize image both pixel wise
    and also in terms of size"""

    _SCOPE = OutputData.IMG

    def __init__(self, mean: float = 0.5, std: float = 0.5):
        """Init

        Args:
            mean (float, optional): pixel mean. Defaults to 0.5.
            std (float, optional): pixel std. Defaults to 0.5.
        """

        super().__init__()

        self.mean = mean
        self.std = std

        # additional transformations
        self.to_pil = transforms.ToPILImage()
        self.tensor_from_pil = transforms.ToTensor()

    def __call__(self, data):

        img = data[self._SCOPE]
        assert img.dtype == torch.float32

        # ------------------- apply image transformations -------------------

        img -= self.mean
        img /= self.std

        # ------------------- image normalization -------------------

        data[self._SCOPE] = img

        return data
