# -*- coding: utf-8 -*-
"""
Created on Jun 06 17:23 2018

@author: Denis Tome'
"""
import numpy as np
import utils
import json

__all__ = [
    'filter_joints',
    'synthetic_to_cmu',
    'standardize_pose',
    'standardize_poses',
    'standardize_synthetic_3d_pose',
    'channel_first_to_channel_last',
    'encode_str_utf8'
]


def standardize_pose(pose, dim=2):
    """
    Standardize pose
    :param pose
    :return: pose in a standardized format (n_joints x dims)
    """
    if type(pose) is not np.ndarray:
        pose = np.array(pose)

    if pose.ndim == 1:
        if dim == 2:
            pose = pose.reshape([-1, 2])
        else:
            assert (pose.shape[0] % 3 == 0)
            pose = pose.reshape([-1, 3])

    assert (pose.ndim == 2)

    if pose.shape[0] in [2, 3]:
        pose = np.transpose(pose, [1, 0])

    return pose


def standardize_poses(poses, dim=2):
    """
    Standardize poses
    :param poses of size (batch_size x undefined)
    :return: poses in a standardized format (batch_size x n_joints x dims)
    """
    if type(poses) is not np.ndarray:
        poses = np.array(poses)

    assert (poses.ndim in [2, 3])
    batch_size = poses.shape[0]

    if poses.ndim == 2:
        if dim == 2:
            assert (poses.shape[1] % 2 == 0)
            poses = poses.reshape([batch_size, -1, 2])
        else:
            assert (poses.shape[1] % 3 == 0)
            poses = poses.reshape([batch_size, -1, 3])
    else:
        if poses.shape[1] in [2, 3]:
            poses = np.transpose(poses, [0, 2, 1])

    return poses


def filter_joints(joints, mask):
    """
    Filter joints according to the masl
    :param joints: format (n_joints x n)
    :param mask: format (n_joints)
    :return: filtered jonits
    """
    if type(joints) is not np.ndarray:
        joints = np.array(joints)

    assert (np.min(joints.shape) in [2, 3])

    if joints.shape[1] not in [2, 3]:
        joints = np.transpose(joints, [1, 0])

    idx_selected = np.where(mask)[0]
    return joints[idx_selected]


def channel_first_to_channel_last(data):
    """
    Convert data from channel first (e.g. batch_size x c x w x h)
    to channel last (e.g. batch_size x w x h x c)
    :param data
    :return: converted data
    """
    if type(data) is not np.ndarray:
        data = np.array(data)

    assert (data.ndim in [3, 4])

    if data.ndim == 3:
        return np.transpose(data, [1, 2, 0])

    # 4 dimensions
    return np.transpose(data, [0, 2, 3, 1])


def synthetic_to_cmu(data, joint_names, order):
    """
    Generate 2D pose compatible with CMU skeleton
    starting from synthetic data
    :param data: (N x C) points
    :param order: order of selection
    :return: (NUM_JOINTS x C) pose
    """
    assert (data.ndim == 2)
    assert (data.shape[0] == len(joint_names))
    pose = np.zeros([len(order), data.shape[1]],
                    dtype=np.float32)
    for jid, j in enumerate(order):
        pose[jid] = data[joint_names[j]]
    return pose


def standardize_synthetic_3d_pose(pose):
    """
    Standardize sythetic 3D poses to have a uniform
    representation as the other datasets.
    :param pose: (NUM_3D_JOINTS x 3)
    :return: standardized pose
    """
    assert ((pose.shape[0] == utils.NUM_JOINTS) or
            (pose.shape[0] == utils.NUM_2D_JOINTS))

    # from cm to m
    pose /= 100
    return pose


def encode_str_utf8(string):
    """
    Convert string in utf8 format
    :param string
    :return: converted string
    """
    return string.encode('utf8')
