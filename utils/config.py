# -*- coding: utf-8 -*-
"""
Common constants used throughout the project

@author: Denis Tome'

"""
import socket
from os.path import dirname, abspath, join
from easydict import EasyDict as edict
from utils.io import ensure_dir

__all__ = [
    'DIRS',
    'MODELS',
    'SETTINGS'
]

_CURR_DIR = dirname(abspath(__file__))

DIRS = edict({
    'root': join(_CURR_DIR, '../'),
    'data': ensure_dir(join(_CURR_DIR, '../data/')),
    'checkpoint': ensure_dir(join(_CURR_DIR, '../data/checkpoints/')),
    'output': ensure_dir(join(_CURR_DIR, '../data/output/'))
})

# ------------------------------ Specific --------------------------------------

MODELS = edict({
    'ae': {
        'z_size': 20,
        'h_l_size': 32
    }
})

# ------------------------------ Generic ---------------------------------------

# machine specific configurations
_HOST_NAME = socket.gethostname()
_MPL_MODE = 'TkAgg'
if socket.gethostname() == 'training':
    _MPL_MODE = 'agg'

SETTINGS = edict({
    'mpl_mode': _MPL_MODE
})
