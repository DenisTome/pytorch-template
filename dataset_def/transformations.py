# -*- coding: utf-8 -*-
"""
Transformations

Each transformation needs to defined the scope
to allow the transformation to be applied only to the
correct piece of information.

@author: Denis Tome'

"""
import torch
from base import FrameworkClass
from dataset_def import OutputData
import utils.math as umath
from utils import skeletons, conversion
from utils import config

__all__ = [
    'Compose',
    'Translation',
    'Rotation',
    'QuaternionToR'
]


class Compose(FrameworkClass):
    """Compose transformations

    This replaces the torch.utils.Compose class
    """

    def __init__(self, transforms):
        """Init"""

        super().__init__()
        self.transforms = transforms
        self._check_transform()

    def _check_transform(self):

        if not isinstance(self.transforms, list):
            self._logger.warning(
                'Comp transformation with single transformation...')
            self.transforms = [self.transforms]
            return

    def __call__(self, data, scope):

        for t in self.transforms:
            if bool(t.get_scope() & scope):
                data = t(data, scope)

        return data[scope]

    def __repr__(self):

        format_string = self.__class__.__name__ + '('

        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)

        format_string += '\n)'
        return format_string


class Transformation(FrameworkClass):
    """Base Trasnformation class"""

    SCOPE = OutputData.P3D

    def __init__(self, d_name):
        """Init"""

        super().__init__()

        assert d_name in config.dataset.supported
        self.d_name = d_name

    def get_scope(self):
        """Get scope, the target of the transformation.
        E.g. scope = OutputData.P3D"""

        return self.SCOPE


class Translation(Transformation):
    """Transformation to class remove root
    translation from poses"""

    SCOPE = OutputData.P3D

    def __init__(self, *args, **kwargs):
        """Init

        Arguments:
            d_name {str} -- dataset name
        """

        super().__init__(*args, **kwargs)

        root_joint_name = config.dataset[self.d_name].root_node
        self.root_id = skeletons[self.d_name].joints[root_joint_name]

    def __call__(self, data, scope):

        sample = data[scope]
        root_p3d = sample[self.root_id]

        # translate
        translated = sample - root_p3d

        data[self.SCOPE] = translated
        return data


class Rotation(Transformation):
    """Transformation class to remove rotation from
    the input poses to have them all aligned in the
    same direction."""

    SCOPE = OutputData.P3D | OutputData.ROT

    def __init__(self, d_name):
        """Init"""

        super().__init__(d_name)

        # joint ids to use for rotation of the poses
        self.rot_norm = skeletons[d_name].normalization.rotation

    def __call__(self, data, scope):

        # set root rotation to zero
        rot = data[OutputData.ROT]
        p3d = data[OutputData.P3D].double()

        # ------------------- target rotation -------------------

        if scope == OutputData.ROT:

            if rot is None:
                rot = torch.zeros([p3d.shape[0], 4])
            else:
                rot[0] = torch.tensor([0, 0, 0, 1])

            data[OutputData.ROT] = rot.float()
            return data

        # ------------------- target position -------------------

        # rotation is give; rotate back pose
        if rot is not None:

            root_rot = rot[0].data.numpy()
            R = umath.quaternion_matrix(root_rot, axes='xyzw')
            R = torch.tensor(R).t()

            rotated = R[:3, :3].mm(p3d.t()).t()
            data[OutputData.P3D] = rotated.float()

            return data

        # compute rotation and rotate back pose
        pose_dir = p3d[self.rot_norm[0]] - p3d[self.rot_norm[1]]

        theta = torch.atan2(pose_dir[1], pose_dir[0])

        rot = torch.tensor(
            [[torch.cos(theta), -torch.sin(theta), 0],
             [torch.sin(theta), torch.cos(theta), 0],
             [0, 0, 1]]
        ).t()

        # R*p^t for new rotations in format 3 x J
        rotated = rot.mm(p3d.t()).t()

        data[OutputData.P3D] = rotated.float()

        return data


class QuaternionToR(Transformation):
    """Transform rotations from quaternion
    representation to SO(3)"""

    SCOPE = OutputData.ROT

    def __call__(self, data, scope):

        rot = data[scope].data.numpy()

        if rot is None:
            return data

        R_j = torch.zeros((rot.shape[0], 3, 3))
        for jid, q_j in enumerate(rot):
            R = umath.quaternion_matrix(q_j, axes='xyzw')[:3, :3]
            R_j[jid] = torch.Tensor(R)

        data[OutputData.ROT] = R_j.float()

        return data
