# -*- coding: utf-8 -*-
"""
Created on Jan 18 16:49 2019

@author: Denis Tome'
"""
import logging

class FrameworkClass(object):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
