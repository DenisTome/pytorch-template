# -*- coding: utf-8 -*-
"""
Utility for conversion related operations

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.2.0"


from math import pi
import torch
import torchgeometry as tgm

_METRIC = {'mm': 1000, 'cm': 100, 'm': 1}
_ORIENTATIONS = {
    'y-up': torch.tensor([pi/2, 0.0, 0.0]).view([1, -1]),
    'y-down': torch.tensor([-pi/2, 0.0, 0.0]).view([1, -1]),
    'z-up': torch.tensor([0.0, 0.0, 0.0]).view([1, -1])
}

__all__ = [
    'match_metric',
    'match_orientation',
    'pixel2cam',
    'cam2pixel',
    'world_to_camera'
]


def match_metric(input_metric: str, target_metric: str) -> float:
    """Convert metric

    Args:
        input_metric (str): metric
        target_metric (str): metric

    Returns:
        float: scale factor
    """

    assert input_metric in list(_METRIC.keys())
    assert target_metric in list(_METRIC.keys())

    scale = _METRIC[target_metric] / _METRIC[input_metric]

    return scale


def match_orientation(input_orientation: str, target_orientation: str) -> torch.Tensor:
    """Generate rotation matrix to match orientation

    Args:
        input_orientation (str): orientation type
        target_orientation (str): orientation type

    Returns:
        torch.Tensor: rotation matrix (4x4)
    """

    r_input = tgm.angle_axis_to_rotation_matrix(
        _ORIENTATIONS[input_orientation])[0]
    r_target = tgm.angle_axis_to_rotation_matrix(
        _ORIENTATIONS[target_orientation])[0]
    rot = r_target.mm(r_input.t())

    return rot


def pixel2cam(pixel_coord, f, c):
    """Pixel to coordinates

    Args:
        pixel_coord (torch.Tensor): pixels
        f (torch.Tensor): focal length
        c (torch.Tensor): camera origin

    Returns:
        Tensor: coordinates in cam coordinate
    """

    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]

    points = torch.cat([x.unsqueeze_(1),
                        y.unsqueeze_(1),
                        z.unsqueeze_(1)], dim=-1)

    return points


def cam2pixel(cam_coord, f, c):
    """From camera coordinates to pixels

    Args:
        cam_coord (torch.Tensor): format (N_JOINTS x 3)
        f (torch.Tensor): focal length
        c (torch.Tensor): original coordinates

    Returns:
        torch.Tensor: u coordinates
        torch.Tensor: v coordinates
        torch.Tensor: z coordinates
    """

    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    z = cam_coord[..., 2]

    return x, y, z


def world_to_camera(joints, R, T, f, c):
    """Project from world coordinates to the camera space

    Args:
        joints (torch.Tensor): p3d with format (N_JOINTS x 3)
        R (torch.Tensor): rotation matrix
        T (torch.Tensor): translation matrix
        f (torch.Tensor): focal length (format [2])
        c (torch.Tensor): optical center (format [2])

    Returns:
        torch.Tensor: joints in pixel coordinates
        torch.Tensor: joints in camera coordinates
    """

    assert T.shape[0] == 3
    assert f.shape[0] == 2
    assert c.shape[0] == 2

    # joints in camera reference system
    n_joints = joints.shape[0]
    joint_cam = torch.zeros(n_joints, 3)

    joint_cam = torch.mm(R, (joints - T).T).T
    # for i in range(n_joints):
    # joint_cam[i] = torch.dot(R, joints[i] - T)

    # joint in pixel coordinates
    joint_px = torch.zeros(n_joints, 3)

    u, v, d = cam2pixel(joint_cam, f, c)
    joint_px[:, 0] = u
    joint_px[:, 1] = v
    joint_px[:, 2] = d

    return joint_px, joint_cam
