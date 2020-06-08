# -*- coding: utf-8 -*-
"""
BVHReader class

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__copyright__ = "Copyright 1998-2020 Epic Games, Inc. All Rights Reserved."
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"


import re
import numpy as np
import utils.math as umath
from .node import BvhNode


class BvhReader:
    """Bvh Class"""

    def __init__(self, data):
        """Init

        Arguments:
            data {File} -- file
        """

        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.tokenize()

    def tokenize(self) -> None:
        """Parse file and extract info stored in nodes"""

        # ------------------- Go through lines -------------------
        first_round = []
        for char in self.data:
            first_round.append(re.split('\\s+', char.strip()))

        # ------------------- Process -------------------
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    def search(self, *items) -> list:
        """Search according to criteria

        Returns:
            list -- list of BvhNodes selected according to criteria
        """

        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)

            # name not matching searching criteria (list)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self) -> list:
        """Get joints

        Returns:
            list -- list of joint names with definition
        """

        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self) -> list:
        """Get joint names

        Returns:
            list -- joint names
        """

        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joint(self, name: str) -> BvhNode:
        """Get joint with name

        Arguments:
            name {str} -- joint name

        Raises:
            LookupError: joint not found

        Returns:
            BvhNode -- joint node
        """

        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]

        raise LookupError('joint not found')

    def joint_direct_children(self, name: str) -> list:
        """Get joints of given joint name

        Arguments:
            name {str} -- joint name to search for

        Returns:
            list -- list of BvhNodes
        """

        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def joint_direct_children_names(self, name: str) -> list:
        """Get joints of given joint name

        Arguments:
            name {str} -- joint name to search for

        Returns:
            list -- list of BvhNodes
        """

        joint = self.get_joint(name)
        return [child for child in joint.get_filter_val('JOINT')]

    def joint_tree_children(self, name: str) -> list:
        """Get all under joint of given name

        Arguments:
            name {str} -- joint name to search for

        Returns:
            list -- list of BvhNodes
        """

        joint = self.get_joint(name)
        if joint.isLeafJoint():
            return []

        level_children = [child for child in joint.get_filter_val('JOINT')]
        children = level_children.copy()
        for child in level_children:
            children.extend(self.joint_tree_children(child))

        return children

    def get_joint_index(self, name: str) -> int:
        """Get joint index

        Arguments:
            name {str} -- joint name

        Returns:
            int -- joint position in the list
        """

        return self.get_joints().index(self.get_joint(name))

    def joint_offset(self, name: str) -> tuple:
        """Get joint offset

        Arguments:
            name {str} -- joint name

        Returns:
            tuple -- (x, y, z) offset
        """

        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name: str) -> list:
        """Get joint channels

        Arguments:
            name {str} -- joint name

        Returns:
            list -- joint channels
        """

        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, name: str) -> int:
        """Get joint channel index

        Arguments:
            name {str} -- joint name

        Raises:
            LookupError: Joint not found

        Returns:
            int -- index of joint channels
        """

        index = 0
        for joint in self.get_joints():
            if joint.value[1] == name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint: str, channel: str) -> int:
        """Get specified joint channel index

        Arguments:
            joint {str} -- joint name
            channel {str} -- channel name

        Returns:
            int -- local channel index
        """

        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index: int,
                            joint: str, channel: str,
                            value=None) -> float:
        """Get joint channel for given frame

        Arguments:
            frame_index {int} -- frame idx
            joint {str} -- joint name
            channel {str} -- channel name

        Keyword Arguments:
            value {str} -- value (default: {None})

        Returns:
            float -- value
        """

        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def set_frame_joint_channel(self, frame_index: int,
                                joint: str, channel: str,
                                value: float):
        """Set joint channel for given frame

        Arguments:
            frame_index {int} -- frame idx
            joint {str} -- joint name
            channel {str} -- channel name
            value {str} -- value (default: {None})
        """

        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        self.frames[frame_index][joint_index + channel_index] = value

    def frame_joint_channels(self, frame_index: int, joint: str,
                             channels: str, value=None) -> list:
        """Get joint channels for frame

        Arguments:
            frame_index {int} -- frame idx
            joint {str} -- joint name
            channels {str} -- channel name

        Keyword Arguments:
            value {obj} -- value (default: {None})

        Returns:
            list -- list of channel values
        """

        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint: str,
                              channels: str, value=None) -> list:
        """Get joint channels for all frames

        Arguments:
            joint {str} -- joint name
            channels {str} -- channel name

        Keyword Arguments:
            value {obj} -- value (default: {None})

        Returns:
            list -- list of channel values
        """

        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name: str) -> str:
        """Joint parent name

        Arguments:
            name {str} -- joint name

        Returns:
            str -- parent name
        """

        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def is_joint_root(self, name: str) -> bool:
        """Get if joint is root

        Arguments:
            name {str} -- joint name

        Returns:
            bool -- true of root joint
        """

        joint = self.get_joint(name)
        if joint.parent == self.root:
            return True

        return False

    def joint_parent_index(self, name: str) -> int:
        """Joint parent index

        Arguments:
            name {str} -- joint name

        Returns:
            int -- joint parent index
        """

        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    def _fk_joint(self, name, fid):
        """Forwark kinematics

        Arguments:
            name {str} -- joint name
            # fid {int} -- frame number

        Returns:
            np.ndarray -- pose
        """

        # ------------------- Static transformations -------------------

        # offset corresponds to the join translation wrt parent node
        offset = self.joint_offset(name)
        stransmat = np.eye(4)
        stransmat[:3, -1] = offset

        if self.is_joint_root(name):
            local_to_world = stransmat
        else:
            parent_trsf = self.joint_parent(name).trsf
            local_to_world = np.dot(parent_trsf, stransmat)

        # ------------------- Dynamic transformations -------------------
        joint_channels = self.joint_channels(name)

        drotmat = np.eye(4)
        for c_name in joint_channels:
            if 'rotation' not in c_name:
                continue

            channel_axis = c_name[0].lower()
            theta = umath.deg_to_rad(
                self.frame_joint_channel(fid, name, c_name))

            R = umath.euler_matrix_from_axis(theta, channel_axis)
            drotmat = np.dot(drotmat, R)

        # ------------------- Joint transformation -------------------
        joint = self.get_joint(name)
        # last operation, to first operation
        trsf = np.dot(local_to_world, drotmat)

        joint.trsf = trsf
        joint.local_trsf = umath.quaternion_from_matrix(drotmat, axes='xyzw')

        # call children transformations
        children = self.joint_direct_children_names(name)
        for child in children:
            self._fk_joint(child, fid)

    def forward_kinematics(self, fid: int):
        """Apply forward kinematics for specific pose at given frame

        Arguments:
            fid {int} -- frame id

        Returns:
            np.ndarray -- pose at frame fid
        """

        assert fid >= 0
        assert fid < self.nframes

        self._fk_joint(self.get_joints_names()[0],
                       fid)

    @property
    def nframes(self):
        """Get number of frames"""
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        """Get frame time"""
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')
