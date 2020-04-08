# -*- coding: utf-8 -*-
"""
Definition of a data sample for lmdb

@author: Denis Tome'

"""


class DataSample():
    """Data sample class"""

    p3d = None
    rot = None
    t = None

    def __init__(self, p3d, rot, t=None):
        """Initialize class

        Arguments:
            p3d {np.ndarray} -- joint positions
            rot {np.ndarray} -- local joint rotations

        Keywork Arguments:
            t {tensor} -- root joint translation (default: {None})
        """

        self.p3d = p3d
        self.rot = rot
        if t is not None:
            self.t = t
