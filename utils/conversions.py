# -*- coding: utf-8 -*-
"""
Created on Jun 06 17:23 2018

@author: Denis Tome'

"""
import numpy as np

__all__ = [
    'synthetic_to_cmu',
    'standardize_pose',
    'standardize_poses',
    'channel_first_to_channel_last',
    'encode_str_utf8'
]


def standardize_pose(pose, dim=2):
    """Standardize pose

    Arguments:
        pose {numpy array | lists} -- format undefined

    Keyword Arguments:
        dim {int} -- number of dimensions (default: {2})

    Returns:
        numpy array -- format (N_JOINTS x dim)
    """

    if not isinstance(pose, np.ndarray):
        pose = np.array(pose)

    if pose.ndim == 1:
        if dim == 2:
            pose = pose.reshape([-1, 2])
        else:
            assert pose.shape[0] % 3 == 0
            pose = pose.reshape([-1, 3])

    assert pose.ndim == 2

    if pose.shape[0] in [2, 3]:
        pose = np.transpose(pose, [1, 0])

    return pose


def standardize_poses(poses, dim=2):
    """Standardize multiple poses

    Arguments:
        poses {numpy array} -- format (N_POSES x N_DIMS)

    Keyword Arguments:
        dim {int} -- number of final dimensions per pose (default: {2})

    Returns:
        numpy array -- format (N_POSES x N_JOINTS x DIM)
    """

    if not isinstance(poses, np.ndarray):
        poses = np.array(poses)

    assert poses.ndim in [2, 3]
    batch_size = poses.shape[0]

    if poses.ndim == 2:
        if dim == 2:
            assert poses.shape[1] % 2 == 0
            poses = poses.reshape([batch_size, -1, 2])
        else:
            assert poses.shape[1] % 3 == 0
            poses = poses.reshape([batch_size, -1, 3])
    else:
        if poses.shape[1] in [2, 3]:
            poses = np.transpose(poses, [0, 2, 1])

    return poses


def channel_first_to_channel_last(data):
    """Put channel in the last dimension

    Arguments:
        data {numpy array} -- tensor

    Returns:
        numpy arrat -- channel is in the last dimension
    """

    if isinstance(data, np.ndarray):
        data = np.array(data)

    assert data.ndim in [3, 4]

    if data.ndim == 3:
        return np.transpose(data, [1, 2, 0])

    # 4 dimensions
    return np.transpose(data, [0, 2, 3, 1])


def synthetic_to_cmu(data, joint_names, order):
    """Convert to CMU skeleton

    Arguments:
        data {numpy array} -- format (N_JOINTS x DIMS)
        joint_names {list} -- joint names
        order {int} -- order of selection

    Returns:
        numpy array -- format (N_CMU_JOINTS x DIMS)
    """

    assert data.ndim == 2
    assert data.shape[0] == len(joint_names)
    pose = np.zeros([len(order), data.shape[1]],
                    dtype=np.float32)
    for jid, j in enumerate(order):
        pose[jid] = data[joint_names[j]]
    return pose


def encode_str_utf8(string):
    """Convert to utf8 encoding

    Arguments:
        string {str} -- content to be converted

    Returns:
        str -- converted string
    """

    return string.encode('utf8')
