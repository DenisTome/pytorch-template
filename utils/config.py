# -*- coding: utf-8 -*-
"""
Configurations used throughout the project

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"
__all__ = [
    'config',
    'model_config',
    'machine',
    'MachineType'
]

from os.path import dirname, abspath, join, exists
from enum import Enum, auto
import socket
import yaml
from easydict import EasyDict as edict
from logger import ConsoleLogger
from utils.io import ensure_dir

_LOGGER = ConsoleLogger('Config')
_CURR_DIR = dirname(abspath(__file__))
_ROOT_DIR = join(_CURR_DIR, '../')


class MachineType(Enum):
    """Machine type"""

    LOCAL = auto()
    AWS = auto()


def load_config() -> dict:
    """Load configuration file

    Returns:
        dict: dictionary with project configuration information
    """

    # path with project configs
    config_path = join(_ROOT_DIR, 'config/general.yml')
    if not exists(config_path):
        raise Exception('File {} does not exist!'.format(config_path))

    with open(config_path) as fin:
        config_data = edict(yaml.safe_load(fin))

    # fix paths wrt project root dir path
    for key, val in config_data.dirs.items():
        config_data.dirs[key] = ensure_dir(abspath(join(_ROOT_DIR, val)))

    config_data.dirs.root = abspath(_ROOT_DIR)

    return config_data


def load_model_config() -> dict:
    """Load configuration file for model

    Returns:
        dict: dictionary with model configuration information
    """

    model_config_path = join(_ROOT_DIR, 'config/model/params.yml')

    with open(model_config_path) as fin:
        model_config_data = edict(yaml.safe_load(fin))

    return model_config_data


def load_machine_config() -> dict:
    """Load configuration file for model

    Returns:
        dict: dictionary with machine configuration information
    """

    machine_path = join(_ROOT_DIR, 'config/machine')
    machine_name = socket.gethostname()

    if 'ip-' in machine_name:
        # EC2 machine
        raise RuntimeError('AWS not supported yet!')

    # Local machine
    machine_config_path = join(machine_path, 'local.yml')
    with open(machine_config_path) as fin:
        machine_config_data = edict(yaml.safe_load(fin))

    machine_config_data.update({'Type': MachineType.LOCAL})

    return machine_config_data


# ------------------------------------------------------
# --------------------- Load parts ---------------------

config = load_config()
model_config = load_model_config()
machine = load_machine_config()
