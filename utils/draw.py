# -*- coding: utf-8 -*-
"""
Draw class for visualizing the different poses

@author: Denis Tome'

"""
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from logger import ConsoleLogger
from utils import config, skeletons, conversion

__all__ = [
    'GLPoseVisualizer',
    'PLT2DVisualizer'
]


class PoseVisualizer():
    """Base class"""

    def __init__(self, line_size=None, marker_size=None):
        """Initialization

        Keyword Arguments:
            line_size {float} -- line size (default: {None})
            marker_size {float} -- marker size (default: {None})
        """

        super().__init__()
        self._logger = ConsoleLogger('PoseVisualizer')

        # ------------------- custom -------------------

        if not line_size:
            self.line_size = config.draw.size.line
        else:
            self.line_size = line_size

        if not marker_size:
            self.marker_size = config.draw.size.marker
        else:
            self.marker_size = marker_size

        self.scale = conversion.match_metric(
            config.model.pose.metric,
            config.draw.metric
        )

    @staticmethod
    def _get_arg(name: str, **kwargs):
        """Get argument from kwargs

        Args:
            name (str): argument name
        """

        if name not in list(kwargs.keys()):
            return None

        return kwargs[name]

    @staticmethod
    def _clip_to_max(image, max_value):
        """Clip image intensities to the maximum
        value defined by max

        Arguments:
            image {numpy array} -- RGB image
            max_value {float | int} -- maximum value

        Returns:
            numpy array -- clipped image
        """

        shape = image.shape
        img = image.flatten()
        idx = np.where(img > max_value)[0]
        img[idx] = max_value

        new_img = img.reshape(shape)
        return new_img


class PLTPoseVisualizer(PoseVisualizer):
    """Class specifing visualization parameters"""

    def __init__(self, legend=None, axis_range=None, **kwargs):
        """Init class

        Keyword Arguments:
            legend {list} -- legend element names;
                             format ['name_a', 'name_b']
                             if None use ['prediction', 'ground truth'] (default: {None})
            max_axis {list} -- axis range
                               format[mix_x, max_x]
                               if None, adapt behaviour on a per frame basis
                               (default: {None})
        """

        super().__init__(**kwargs)

        # ------------------- Properties -------------------

        self.set_legend(legend)

        if not axis_range:
            self.axis_range = None
        else:
            try:
                assert isinstance(axis_range, list)
                assert len(axis_range) == 2
                assert all(isinstance(x, float) for x in axis_range)
            except AssertionError:
                self._logger.error('Unexpected format for axis raanges!')

            self.axis_range = axis_range

    def _scale_plot(self, pose, ax):
        """Scale plot according to data

        Arguments:
            pose {numpy array} -- 2D or 3D pose
            ax {ax} -- ax contained in figure
        """

        p3d = pose['data'][:, :3] * self.scale

        if self.axis_range is None:
            val = np.max([np.abs(p3d.min()), np.abs(p3d.max())])
            smallest = - val
            largest = val
        else:
            smallest = self.axis_range[0]
            largest = self.axis_range[1]

        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)

    def _check_pose(self, pose) -> None:
        """Check pose format

        Arguments:
            pose {dict} -- pose
        """

        if not isinstance(pose, dict):
            self._logger.error('Pose parameter needs to be a dictionary!')

        if not pose['d_type'] in config.dataset.supported:
            self._logger.error('Pose not supported!')

        if pose['data'].shape[0] != skeletons[pose['d_type']].n_joints:
            self._logger.error('Pose has wrong dimensionality!')

    def _add_pose(self, pose, ax, color=None):
        """Add pose

        Arguments:
            pose {dict} -- pose ['d_type', 'data']

        Keyword Arguments:
            color {list} -- rgba color (default: {None})
        """
        self._check_pose(pose)

        p3d = pose['data'][:, :3] * self.scale

        conn = skeletons[pose['d_type']].limbs.connections
        col_id = skeletons[pose['d_type']].limbs.color_id

        if not color:
            limb_color = skeletons[pose['d_type']].colors
        else:
            if not isinstance(color, list):
                self._logger.error('Color format is wrong!')

            limb_color = [color] * p3d.shape[0]

        # ------------------- plot limbs -------------------
        for lid, limb in enumerate(conn):

            pts = np.vstack([p3d[limb[0]], p3d[limb[1]]])

            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=limb_color[col_id[lid]],
                    linewidth=self.line_size)

        # ------------------- plot joints -------------------

        colors = np.ones([p3d.shape[0], 4]) * config.draw.colors.joints
        ax.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2],
                   c=colors,
                   marker='o',
                   edgecolor=colors,
                   s=config.draw.size.marker)

    def _legend(self, ax):
        """Add legend"""

        custom_lines = [
            Line2D([0], [0], color=config.draw.colors.pose.prediction, lw=4),
            Line2D([0], [0], color=config.draw.colors.pose.ground_truth, lw=4)
        ]
        ax.legend(custom_lines, self.legend)

    def set_legend(self, legend: list) -> None:
        """Set legend

        Arguments:
            legend {list} -- legend
        """

        if legend is None:
            self.legend = ['Pred', 'Gt']
            return

        try:
            assert isinstance(legend, list)
            assert len(legend) == 2
            assert all(isinstance(x, str) for x in legend)
        except AssertionError:
            self._logger.error('Unexpected format for legend parameter!')

        self.legend = legend

    def plot3DPose(self, pose, save_image=False, **kwargs):
        """Plot 3D pose

        Arguments:
            pose {dict} -- ['d_type', 'data']

        Keyword Arguments:
            save_image {bool} -- return image instead of plotting (default: {False})

        Returns:
            np.ndarray -- rgb information if image is True
        """

        # generate figure
        fig = self._get_arg('fig', **kwargs)
        if not fig:
            fig = plt.figure(num=None, figsize=(8, 8),
                             dpi=100, facecolor='w', edgecolor='k')

        ax = self._get_arg('ax', **kwargs)
        if not ax:
            ax = fig.add_subplot(111, projection='3d')

        title = self._get_arg('title', **kwargs)
        if title:
            ax.set_title(title)

        self._add_pose(pose, ax)
        self._scale_plot(pose, ax)

        fig.canvas.draw()

        if save_image:
            w, h = fig.canvas.get_width_height()
            save_image = np.fromstring(fig.canvas.tostring_rgb(),
                                       dtype='uint8')
            save_image = np.reshape(save_image, [h, w, 3])

            plt.close(fig)
            return save_image

        return fig

    def plot3DPoses(self, pose_a, pose_b, save_image=False, **kwargs):
        """Plot 3D pose

        Arguments:
            pose_a {dict} -- ['d_type', 'data']
            pose_b {dict} -- ['d_type', 'data']

        Keyword Arguments:
            save_image {bool} -- return image instead of plotting (default: {False})

        Returns:
            np.ndarray -- rgb information if save_image is True
        """

        # generate figure
        fig = self._get_arg('fig', **kwargs)
        ax = self._get_arg('ax', **kwargs)
        if not fig:
            if ax:
                raise AssertionError('if ax is given, so do fig!')

            fig = plt.figure(num=None, figsize=(8, 8),
                             dpi=100, facecolor='w', edgecolor='k')

        if not ax:
            ax = fig.add_subplot(111, projection='3d')

        self._add_pose(pose_a,
                       ax,
                       config.draw.colors.pose.prediction)
        self._add_pose(pose_b,
                       ax,
                       config.draw.colors.pose.ground_truth)
        self._scale_plot(pose_b, ax)
        self._legend(ax)

        fig.canvas.draw()

        if save_image:
            w, h = fig.canvas.get_width_height()
            image = np.fromstring(fig.canvas.tostring_rgb(),
                                  dtype='uint8')
            image = np.reshape(image, [h, w, 3])

            plt.close(fig)
            return image

        return fig

    @staticmethod
    def plotEmbeddings(embedding, dataset_idx, d_names, save_image=False):
        """Plot pose embeddings

        Arguments:
            embedding {np.ndarray} -- projected embedding format (N x 2)
            dataset_idx {list} -- dataset idx per frame
            d_names {lsit} -- dataset names

        Keyword Arguments:
            save_image {bool} -- returning image (default: {False})

        Returns:
            np.ndarray -- image
        """

        # generate figure
        fig = plt.figure(num=None, figsize=(8, 8),
                         dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(111)

        start = 0
        col = ['red',
               'blue']
        for idx, d_size in enumerate(dataset_idx):
            plt.plot(
                embedding[start:start+d_size, 0],
                embedding[start:start+d_size, 1],
                'o', color=col[idx])
            start += d_size

        ax.legend(d_names)

        fig.canvas.draw()

        image = np.empty([])
        if save_image:
            w, h = fig.canvas.get_width_height()
            image = np.fromstring(fig.canvas.tostring_rgb(),
                                  dtype='uint8')
            image = np.reshape(image, [h, w, 3])

            plt.close(fig)
            return image

        plt.show()
        return image


class PLT2DVisualizer(PoseVisualizer):
    """Class specifing visualization parameters"""

    def _check_pose(self, pose) -> None:
        """Check pose format

        Arguments:
            pose {dict} -- pose
        """

        if not isinstance(pose, dict):
            self._logger.error('Pose parameter needs to be a dictionary!')

        if not pose['d_type'] in config.dataset.supported:
            self._logger.error('Pose not supported!')

        if pose['data'].shape[0] != skeletons[pose['d_type']].n_joints:
            self._logger.error('Pose has wrong dimensionality!')

    def plotImageJoints(self, image, p2d, save_image=False, **kwargs):
        """Plot image with 2d joint positions

        Arguments:
            image {np.ndarray} -- image
            p2d {np.ndarray} -- joint positions

        Keyword Arguments:
            save_image {bool} -- return image instead of plotting (default: {False})

        Returns:
            np.ndarray -- rgb information if save_image is True
        """

        c = self._get_arg('color', **kwargs)
        if c is None:
            c = (255, 255, 0)

        r = self._get_arg('radius', **kwargs)
        if r is None:
            r = 2

        if image.dtype == np.float32:
            image *= 255.0
            image = image.astype(np.uint8)

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        # draw joint positions
        for j in p2d:
            draw.ellipse((j[0]-r, j[1]-r, j[0]+r, j[1]+r), fill=c)

        if save_image:
            return np.asarray(img)

        img.show()

    def plotImage(self, image, **kwargs):
        """Plot image

        Arguments:
            image {np.ndarray} -- image

        Returns:
            np.ndarray -- rgb information if save_image is True
        """

        if image.dtype == np.float32:
            image *= 255.0
            image = image.astype(np.uint8)

        img = Image.fromarray(image)
        img.show()
