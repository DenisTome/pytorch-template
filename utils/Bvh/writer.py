# -*- coding: utf-8 -*-
"""
BvhWriter class

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__copyright__ = "Copyright 1998-2020 Epic Games, Inc. All Rights Reserved."
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"


import os
import numpy as np
import utils.math as umath
from utils import config, skeletons
from .reader import BvhReader


class BvhWriter:
    """Bvh writer class"""

    _PATH = 'config/dataset/skeleton_few.bvh'

    def __init__(self, rotations_local=False, dataset_name_format='cmu',
                 frame_rate: float = None):

        assert dataset_name_format in config.dataset.supported

        if dataset_name_format != 'cmu':
            raise RuntimeError('CMU Bvh format is the only supported one!')

        # ------------------- io related -------------------
        self.bvh_str, self.bvh = self._load_bvh()

        # ------------------- config -------------------
        self.d_name = dataset_name_format
        self.skel_def = skeletons[self.d_name].joints

        # ------------------- bvh related -------------------
        self.n_channels = self._get_n_channels()
        self.frames = []
        self.n_frames = 0
        if frame_rate is None:
            self.frate_rate = 1.0 / 30.0
        else:
            self.frame_rate = frame_rate

        self.rotations_local = rotations_local

    def _load_bvh(self):
        """Load bvh file

        Returns:
            str -- bvh string
        """

        generic_skel = os.path.join(config.dirs.data, self._PATH)

        with open(generic_skel) as f:
            bvh_reader = BvhReader(f)
            f.seek(0)
            data = f.read()

        return data, bvh_reader

    def _get_n_channels(self):
        """Get total number of channels"""

        def node_channel(data, name):

            n_channels = len(data.joint_channels(name))
            children = data.joint_direct_children_names(name)
            for child in children:
                n_channels += node_channel(data, child)
            return n_channels

        tot_channels = node_channel(
            self.bvh,
            self.bvh.get_joints_names()[0])

        return tot_channels

    def set_frame_rate(self, frame_rate: float):
        """Set frame rate

        Arguments:
            frame_rate {float} -- frame rate
        """

        assert frame_rate > 0.0
        self.frate_rate = frame_rate

    def _add_joint_to_frame(self, frame_data, pose, joint):
        """Place joint rotation in the right posiiton according
        to Bvh definition

        Arguments:
            frame_data {np.ndarray} -- verctor containing frame data
            pose {n.ndarray} -- joint rotations
            joint {str} -- joint name

        Returns:
            np.ndarray -- frame data
        """

        if joint not in list(self.skel_def.keys()):
            return frame_data

        jid = self.skel_def[joint]

        joint_rot = pose[jid]

        # ------------------- joint config -------------------

        j_channels = self.bvh.joint_channels(joint)
        rot_order = [n.replace('rotation', '').lower()
                     for n in j_channels if 'rotation' in n]
        r_type = 'r{}{}{}'.format(*(rot_order))

        # ------------------- euler angles -------------------

        euler_rad = umath.euler_from_matrix(joint_rot, axes=r_type)
        euler_deg = umath.rad_to_deg(np.array(euler_rad))

        # place rotations in the right order at the correct location
        rid = 0
        channel_start_index = self.bvh.get_joint_channels_index(joint)
        for cid, channel in enumerate(j_channels):

            if 'position' in channel:
                continue

            frame_data[channel_start_index + cid] = euler_deg[rid]
            rid += 1

        # ------------------- children -------------------

        children = self.bvh.joint_direct_children_names(joint)
        for child in children:
            frame_data = self._add_joint_to_frame(frame_data, pose, child)

        return frame_data

    def add_frame(self, pose):
        """Add frame to Bvh

        Arguments:
            pose {np.ndarray} -- pose as joint rotations
        """

        assert pose.ndim == 3
        assert pose.shape[0] == len(self.skel_def)

        # initialize
        frame_data = self._add_joint_to_frame(
            np.zeros(self.n_channels, dtype=np.float32),
            pose,
            list(self.skel_def.keys())[0]
        )

        self.frames.append(frame_data)
        self.n_frames += 1

    def save(self, file_path: str):
        """Save to Bvh output file

        Arguments:
            file_path {str} -- file path
        """

        if '.bvh' not in file_path:
            file_path += '.bvh'

        with open(file_path, 'w') as f:

            # write skeleton definition
            f.writelines(self.bvh_str)

            # frames and frame rate
            f.writelines('Frames: {:d}\n'.format(self.n_frames))
            f.writelines('Frame Time: {:f}\n'.format(self.frate_rate))

            for frame in self.frames:
                frame_str = np.array2string(
                    frame,
                    max_line_width=np.inf)
                frame_str = frame_str.replace('[ ', '')
                frame_str = frame_str.replace(']', '')

                f.writelines(frame_str)
                f.writelines('\n')
