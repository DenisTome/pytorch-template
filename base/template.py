# -*- coding: utf-8 -*-
"""
Framework class to be used to extend all
the other classes

@author: Denis Tome'

"""
from logger.console_logger import ConsoleLogger

class FrameworkClass:
    """Framework Class"""

    def __init__(self):
        super().__init__()
        self._logger = ConsoleLogger(self.__class__.__name__)
