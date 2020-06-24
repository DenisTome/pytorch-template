# -*- coding: utf-8 -*-
"""
Transformations

@author: Denis Tome'

"""
import torch
from torchvision import transforms
from base import FrameworkClass
from base.base_dataset import OutputData
import utils.math as umath
from utils.config import skeletons, conversion, config

__all__ = [
    'Compose',
    'Translation',
    'Align',
    'Rotation',
    'QuaternionToR',
    'ImageNormalization'
]


class Compose(FrameworkClass):
    """Compose transformations

    This replaces the torch.utils.Compose class
    """

    def __init__(self, list_trsf: list):
        """List transformations

        Args:
            list_trsf (list): transformations
        """

        super().__init__()
        self.transforms = list_trsf
        self._check_transform()

    def _check_transform(self):
        """Check transformation"""

        if not isinstance(self.transforms, list):
            self._logger.warning(
                'Comp transformation with single transformation...')
            self.transforms = [self.transforms]
            return

    def __call__(self, data: dict, scope: OutputData):
        """Run

        Args:
            data (dict): key are the type of data (based on scope)
            scope (OutputData): data we are interest to process
                                e.g. OutputData.IMG | OutputData.P3D

        Returns:
            dict: transformed data
        """

        for t in self.transforms:
            if bool(t.get_scope() & scope):
                data = t(data)

        return data

    def __repr__(self) -> str:

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

    def __call__(self, data):

        sample = self.to_tensor(data[self.SCOPE])
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

    def __call__(self, data):

        sample = data[self.SCOPE]
        if sample.dim() > 2:
            return self._batch_call(data)

        padd = torch.ones((sample.shape[0], 1))

        # homogeneous coordinates
        p4d = torch.cat([sample, padd], dim=1)

        rotated = self.rot.mm(p4d.t()).t()

        # only xyz components
        scaled = rotated[:, :3] * self.scale

        data[self.SCOPE] = scaled
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

    def __call__(self, data):

        # set root rotation to zero
        rot = self.to_tensor(data[OutputData.ROT])
        p3d = self.to_tensor(data[OutputData.P3D]).double()

        # ------------------- target rotation -------------------

        if OutputData.ROT in data.keys():

            if rot is None:
                rot = torch.zeros([p3d.shape[0], 4])
            else:
                rot[0] = torch.tensor([0, 0, 0, 1])

            data[OutputData.ROT] = rot.float()

        # ------------------- target position -------------------

        if OutputData.P3D in data.keys():

            # rotation is give; rotate back pose
            if rot is not None:

                root_rot = rot[0].data.numpy()
                R = umath.quaternion_matrix(root_rot, axes='xyzw')
                R = torch.tensor(R).t()

                rotated = R[:3, :3].mm(p3d.t()).t()
                data[OutputData.P3D] = rotated.float()
            else:
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

    def __call__(self, data):

        rot = data[self.SCOPE].data.numpy()

        if rot is None:
            return data

        R_j = torch.zeros((rot.shape[0], 3, 3))
        for jid, q_j in enumerate(rot):
            R = umath.quaternion_matrix(q_j, axes='xyzw')[:3, :3]
            R_j[jid] = torch.Tensor(R)

        data[self.SCOPE] = R_j.float()

        return data


class ImageNormalization(Transformation):
    """Transformation class to normalize image both pixel wise
    and also in terms of size"""

    SCOPE = OutputData.IMG

    def __init__(self, d_name: str, mean: float = 0.5, std: float = 0.5):
        """Init

        Args:
            d_name (str): dataset name
            mean (float, optional): pixel mean. Defaults to 0.5.
            std (float, optional): pixel std. Defaults to 0.5.
        """

        super().__init__(d_name)

        self.mean = mean
        self.std = std

        # additional transformations
        self.to_pil = transforms.ToPILImage()
        self.tensor_from_pil = transforms.ToTensor()

    def __call__(self, data):

        img = data[OutputData.IMG]
        assert img.dtype == torch.float32

        # ------------------- apply image transformations -------------------

        img -= self.mean
        img /= self.std

        # ------------------- image normalization -------------------

        data[OutputData.IMG] = img

        return data
