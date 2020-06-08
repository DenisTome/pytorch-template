# -*- coding: utf-8 -*-
"""
Transformations

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
    'Align',
    'Rotation',
    'QuaternionToR'
]


class Compose(FrameworkClass):
    """Compose transformations"""

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

        align_pos = -1
        for tid, t in enumerate(self.transforms):
            if isinstance(t, Align):
                align_pos = tid

        if align_pos < 0:
            return

        if align_pos != (len(self.transforms) - 1):
            self._logger.error('Align transformation must be last!')

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

    @staticmethod
    def to_tensor(data):
        """Return tensor from np.ndarray
        input format"""

        if data is None:
            return data

        if isinstance(data, torch.Tensor):
            return data

        return torch.Tensor(data)


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

        sample = self.to_tensor(data[scope])
        root_p3d = sample[self.root_id]

        # translate
        translated = sample - root_p3d

        data[self.SCOPE] = translated
        return data


class Align(Transformation):
    """Transformation class to align pose with standard convention
    making sure that orientation and scale are correct"""

    SCOPE = OutputData.P3D

    def __init__(self, d_name, cuda=False, batch_size=None):
        """Init

        Arguments:
            d_name {str} -- dataset name

        Keywork Arguments:
            cuda {bool} -- cuda

        """

        super().__init__(d_name)

        self.cuda = cuda
        input_metric = config.dataset[d_name].metric
        input_orientation = config.dataset[d_name].orientation

        self.scale = conversion.match_metric(
            input_metric,
            config.model.pose.metric
        )

        # match desired dataset orientation
        rot = conversion.match_orientation(
            input_orientation,
            config.model.pose.orientation
        )

        if batch_size is not None:
            rot = rot.view(1, 4, 4).repeat(batch_size, 1, 1)

        if cuda:
            self.rot = rot.cuda()
        else:
            self.rot = rot

    def _batch_call(self, sample):
        """Appy trasnformatino to mini-batch"""

        padd = torch.ones((sample.shape[0],
                           sample.shape[1],
                           1))

        if self.cuda:
            padd = padd.cuda()

        # homogeneous coordinates
        p4d = torch.cat([sample, padd], dim=2)

        rotated = torch.bmm(
            self.rot[:p4d.shape[0]],
            p4d.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # only xyz components
        scaled = rotated[:, :, :3] * self.scale

        return scaled

    def __call__(self, data, scope=None):

        if scope is None:
            return self._batch_call(data)

        sample = data[scope]
        padd = torch.ones((sample.shape[0], 1))

        # homogeneous coordinates
        p4d = torch.cat([sample, padd], dim=1)

        rotated = self.rot.mm(p4d.t()).t()

        # only xyz components
        scaled = rotated[:, :3] * self.scale

        data[scope] = scaled
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
        rot = self.to_tensor(data[OutputData.ROT])
        p3d = self.to_tensor(data[OutputData.P3D]).double()

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
