# -*- coding: utf-8 -*-
"""
Basic transformation

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = 'v0.2.0'

from .template import FrameworkClass
from .base_dataset import OutputData


class ComposeTransformations(FrameworkClass):
    """Compose transformations

    This replaces the torch.utils.Compose class
    """

    def __init__(self, list_trsf: list):
        """List transformations

        Args:
            list_trsf (list): transformations
        """

        super().__init__()
        self.transforms = list_trsf
        self._check_transform()

    def _check_transform(self):
        """Check transformation"""

        if not isinstance(self.transforms, list):
            self._logger.warning(
                'Comp transformation with single transformation...')
            self.transforms = [self.transforms]
            return

    def __call__(self, data: dict, scope: OutputData):
        """Run

        Args:
            data (dict): key are the type of data (based on scope)
            scope (OutputData): data we are interest to process
                                e.g. OutputData.IMG | OutputData.P3D

        Returns:
            dict: transformed data
        """

        for t in self.transforms:
            if bool(t.get_scope() & scope):
                data = t(data)

        return data

    def __repr__(self) -> str:

        format_string = self.__class__.__name__ + '('

        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)

        format_string += '\n)'
        return format_string


class BaseTransformation(FrameworkClass):
    """Base Trasnformation class"""

    _SCOPE = OutputData.NONE

    @property
    def scope(self) -> OutputData:
        """Get scope, the target of the transformation

        Returns:
            OutputData: scope of the current transformation
        """

        if self._SCOPE == OutputData.NONE:
            self._logger.error(
                'Scope for the current transformation is not initialized!')

        return self._SCOPE
