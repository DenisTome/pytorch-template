# -*- coding: utf-8 -*-
"""
Common constants used throughout the project

@author: Denis Tome'

"""
import socket
from os.path import dirname, abspath, join
from utils.util import ensure_dir

__all__ = [
    'DATA_DIR',
    'CHKP_DIR',
    'MPL_MODE',
    'OUT_DIR',
    'INFO_DIR'
]

# defining relative paths
CURR_DIR = dirname(abspath(__file__))
ROOT_DIR = join(CURR_DIR, '../')
DATA_DIR = ensure_dir(join(ROOT_DIR, 'data/'))
CHKP_DIR = ensure_dir(join(DATA_DIR, 'checkpoints/'))
OUT_DIR = ensure_dir(join(DATA_DIR, 'output'))
INFO_DIR = ensure_dir(join(DATA_DIR, 'info'))

# ------------------------------ Generic ---------------------------------------

# machine specific configurations
HOST_NAME = socket.gethostname()
MPL_MODE = 'TkAgg'
if socket.gethostname() == 'training':
    MPL_MODE = 'agg'
