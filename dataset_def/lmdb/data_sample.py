# -*- coding: utf-8 -*-
"""
Example of definition of a data sample for lmdb.

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
import numpy as np


class DataSample():
    """Data sample class"""

    # ------------------------------------------------------
    # This depends on the type of data stored in the
    # dataset as well as metadata we want to include
    # in the lmdb version.
    # ------------------------------------------------------

    p3d = None
    rot = None
    t = None

    def __init__(self, p3d: np.array, rot: np.array, t: np.array = None):
        """Initialize class

        Args:
            p3d (np.ndarray): joint positions
            rot (np.ndarray): local joint rotations

        Keywork Arguments:
            t {tensor} -- root joint translation (default: {None})
        """

        self.p3d = p3d
        self.rot = rot
        if t is not None:
            self.t = t
