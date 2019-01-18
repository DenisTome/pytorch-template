# -*- coding: utf-8 -*-
"""
Created on Jun 08 15:16 2018

@author: Denis Tome'
"""
import re
import os
import json
import h5py
import utils
import numpy as np

__all__ = [
    'get_checkpoint',
    'get_filename_from_path',
    'file_exists',
    'write_json',
    'read_from_json',
    'abs_path',
    'get_dir',
    'remove_files',
    'write_h5',
    'read_h5',
    'metadata_to_json',
    'json_to_metadata'
]


def get_checkpoint(resume_dir, epoch=None, iteration=None):
    """
    Retrieve checkpoint
    :param resume_dir: directory with saved models
    :param epoch: if not specified is the last one
    :param iteration: if not specified is the last one
    :return: path
    """
    models_list = [
        f for f in os.listdir(resume_dir) if f.endswith(".pth.tar")
    ]
    models_list.sort()

    if not models_list:
        raise IOError('Directory {} does not contain any model'.format(resume_dir))

    model_name = models_list[-2]
    if epoch is not None:
        # getting the right model
        r = re.compile("ckpt_eph{:02d}_iter.*".format(epoch))
        model_name = [
            m.group(0) for l in models_list for m in [r.search(l)] if m
        ]
        model_name = model_name[0]

        if iteration is not None:
            r = re.compile("ckpt_eph{:02d}_iter{:06d}_.*".format(epoch, iteration))
            model_name = [
                m.group(0) for l in models_list for m in [r.search(l)] if m
            ]
            model_name = model_name[0]

        if not model_name:
            raise Exception('Model {}/ckpt_eph{:02d}_iter{:06d} does not exist'.format(
                resume_dir, epoch, iteration))

    return os.path.join(resume_dir, model_name)


def get_filename_from_path(path):
    """
    Get the filename given a path
    :param path
    :return: filename without format, format
    """
    file_with_format = os.path.split(path)[-1]
    file_format = re.findall(r'\.[a-zA-Z]+', file_with_format)[-1]
    file_name = file_with_format.replace(file_format, '')
    return file_name, file_format


def file_exists(path):
    """
    Check if the file exists
    :param path
    :return: bool
    """
    return os.path.exists(path)


def write_json(file_path, info):
    """
    Save to json file
    :param file_path
    :param info: data
    """
    with open(file_path, 'w') as out_file:
        json.dump(info, out_file, indent=2)


def read_from_json(file_path):
    """
    Retrieve from json file
    :param file_path
    """
    with open(file_path, 'r') as in_file:
        data = json.load(in_file)

    return data


def abs_path(path):
    """
    Get absolute path
    """
    if path:
        return os.path.expanduser(path)

    return None


def get_dir(path):
    """
    Get directory from path
    """
    if '.' in path[-4:]:
        name = path.split('/')[-1]
        dir_name = path.replace('/{}'.format(name), '')
        return dir_name

    return path


def remove_files(paths):
    """
    Remove files given their path
    :param paths: list of paths
    """
    for path in paths:
        os.remove(path)


def write_h5(path, data):
    """
    Save data in h5 file format
    :param path: output file path
    :param data: dictionary
    """
    if '.h5' not in path[-3:]:
        path += '.h5'

    hf = h5py.File(path, 'w')

    if isinstance(data, dict):
        for k, v in data.items():
            if type(v[0]) == str:
                v = [a.encode('utf8') for a in v]
            hf.create_dataset(k, data=v)
    elif isinstance(data, np.ndarray):
        hf.create_dataset('val', data=data)
    else:
        raise NotImplementedError
    hf.close()


def read_h5(path):
    """Load data"""
    if not os.path.isfile(path):
        raise FileNotFoundError()

    data_files = dict()
    h5_data = h5py.File(path)
    tags = list(h5_data.keys())
    for tag in tags:
        tag_data = np.asarray(h5_data[tag]).copy()
        data_files.update({tag: tag_data})
    h5_data.close()

    return data_files


def json_to_metadata(file_path):
    """
    Retrieve metadata from json file
    :param file_path
    """
    return utils.read_from_json(file_path)


def metadata_to_json(file_path, info):
    """
    Save metadata to json file
    :param file_path
    :param info: data
    """
    utils.write_json(file_path, info)