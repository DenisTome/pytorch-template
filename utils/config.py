# -*- coding: utf-8 -*-
"""
Configurations used throughout the project

@author: Denis Tome'

"""
from os.path import dirname, abspath, join
from enum import Enum, auto
import socket
import yaml
from easydict import EasyDict as edict
import numpy as np
import matplotlib as mpl
from logger import ConsoleLogger
from utils.io import ensure_dir

_LOGGER = ConsoleLogger('Config')
_CURR_DIR = dirname(abspath(__file__))
_ROOT_DIR = join(_CURR_DIR, '../')

__all__ = [
    'config',
    'skeletons',
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

    return config_data


def laod_skelton_definitions(configuration: dict) -> dict:
    """Load skeleton definitions and customizations

    Arguments:
        config {dict} -- configuration dic

    Returns:
        edit -- skeleton definitions
    """

    def _check(skel, name):

        if skel.definition.n_joints != len(list(skel.definition.joints.keys())):
            _LOGGER.error('Bad definition of {} skeleton joints!'.format(name))

        if skel.definition.n_limbs != len(skel.definition.limbs.connections):
            _LOGGER.error('Bad definition of {} skeleton limbs!'.format(name))

        n_max = np.max(skel.definition.limbs.color_id) + 1
        if len(skel.definition.colors) != n_max:
            _LOGGER.error('Bad definition of {} skeleton colors!'.format(name))

    dataset_config_path = join(configuration.dirs.data,
                               'config/dataset')

    # load skeleton definitions automatically
    # based on the supported datasets
    supp_datasets = configuration.dataset.supported
    skeleton = {}

    for supp_dataset in supp_datasets:
        skel_path = join(dataset_config_path,
                         '{}_skeleton.yml'.format(supp_dataset))
        with open(skel_path) as fin:
            skel = edict(yaml.safe_load(fin))

        _check(skel, supp_dataset)
        skeleton[supp_dataset] = skel

    return skeleton


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

    def custom_env(mode: str) -> None:
        """Cutom configurations"""

        assert mode in ['agg', 'TkAgg']
        if configuration.status.debug:
            mpl.use('TkAgg')
        else:
            mpl.use(mode)

    def replace_placeholder(path: str, machine_config: dict) -> str:
        """Filter path if there are placeholders"""
        
        if '${dataset_type}' in path:
            path = path.replace('${dataset_type}', machine_config.datasets.type)

        return path

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

    custom_env(machine_config_data.mpl.mode)
    return machine_config_data


# ------------------------------------------------------
# --------------------- Load parts ---------------------
config = load_config()
skeletons = laod_skelton_definitions(config)
model_config = load_model_config(config)
machine = load_machine_config(config)
