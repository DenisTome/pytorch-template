# -*- coding: utf-8 -*-
"""
Framework class to be used to extend all
the other classes

@author: Denis Tome'

"""
import logging

class FrameworkClass:
    """Framework Class"""

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
