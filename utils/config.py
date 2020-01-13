# -*- coding: utf-8 -*-
"""
Configurations used throughout the project

@author: Denis Tome'

"""
from os.path import dirname, abspath, join
import yaml
from easydict import EasyDict as edict
from utils.io import ensure_dir


__all__ = [
    'config',
]


def load_config():
    """Load configuration file

    Returns:
        dict -- dictionary with project configuration information
    """

    curr_dir = dirname(abspath(__file__))
    root_dir = join(curr_dir, '../')

    # path with project configs
    config_path = join(root_dir, 'data/config.yml')

    with open(config_path) as fin:
        config_data = edict(yaml.safe_load(fin))

    # fix paths wrt project root dir path
    for key, val in config_data.dirs.items():
        config_data.dirs[key] = ensure_dir(abspath(join(root_dir, val)))

    return config_data


config = load_config()
