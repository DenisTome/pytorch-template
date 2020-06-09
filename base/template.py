# -*- coding: utf-8 -*-
"""
Framework class to be used to extend all
the other classes

@author: Denis Tome'

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"

from logger.console_logger import ConsoleLogger

class FrameworkClass:
    """Framework Class"""

    def __init__(self):
        super().__init__()
        self._logger = ConsoleLogger(self.__class__.__name__)
