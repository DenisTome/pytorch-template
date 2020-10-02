# -*- coding: utf-8 -*-
"""
Configurations used throughout the project

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
from os.path import dirname, abspath, join
from enum import Enum, auto
import socket
import yaml
from easydict import EasyDict as edict
from logger import ConsoleLogger
from utils.io import ensure_dir

_LOGGER = ConsoleLogger('Config')
_CURR_DIR = dirname(abspath(__file__))
_ROOT_DIR = join(_CURR_DIR, '../')

__all__ = [
    'config',
    'model_config',
    'machine',
    'MachineType'
]


class MachineType(Enum):
    """Machine type"""

    LOCAL = auto()
    AWS = auto()


def load_config() -> dict:
    """Load configuration file

    Returns:
        dict -- dictionary with project configuration information
    """

    # path with project configs
    config_path = join(_ROOT_DIR, 'data/config/general.yml')

    with open(config_path) as fin:
        config_data = edict(yaml.safe_load(fin))

    # fix paths wrt project root dir path
    for key, val in config_data.dirs.items():
        config_data.dirs[key] = ensure_dir(abspath(join(_ROOT_DIR, val)))

    config_data.dirs.root = abspath(_ROOT_DIR)

    return config_data


def load_model_config(configuration: dict) -> dict:
    """Load configuration file for model

    Returns:
        dict -- dictionary with model configuration information
    """

    model_config_path = join(configuration.dirs.data,
                             'config/model/params.yml')

    with open(model_config_path) as fin:
        model_config_data = edict(yaml.safe_load(fin))

    return model_config_data


def load_machine_config(configuration: dict) -> dict:
    """Load configuration file for model

    Returns:
        dict -- dictionary with machine configuration information
    """

    machine_path = join(configuration.dirs.data,
                        'config/machine')

    machine_name = socket.gethostname()

    if 'ip-' in machine_name:
        # EC2 machine
        raise RuntimeError('AWS not supported yet!')
    else:
        # Local machine
        machine_config_path = join(machine_path, 'local.yml')
        with open(machine_config_path) as fin:
            machine_config_data = edict(yaml.safe_load(fin))

        machine_config_data.update({'Type': MachineType.LOCAL})

    return machine_config_data


# ------------------------------------------------------
# --------------------- Load parts ---------------------

config = load_config()
model_config = load_model_config(config)
machine = load_machine_config(config)
