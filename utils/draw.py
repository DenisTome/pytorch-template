# -*- coding: utf-8 -*-
"""
Created on Jun 06 17:42 2018

@author: Denis Tome'
"""
import cv2
import utils
import numpy as np
import matplotlib as mpl
mpl.use(utils.MPL_MODE)
import matplotlib.pyplot as plt

__all__ = [
    'draw_skeleton',
    'plot_pose',
    'plot_poses',
]

_C_LIMBS_IDX = [0, 1, 2, 10,
                0, 3, 4,
                5, 6, 7, 7,
                5, 8, 9, 9]

_COLORS = [[0, 0, 255], [0, 100, 0], [0, 255, 0], [0, 165, 255], [0, 255, 255],
           [255, 255, 0], [100, 0, 0], [255, 0, 0], [130, 0, 75], [255, 0, 255], [0, 0, 0]]


def draw_skeleton(image, pose_2d, p_size=3,
                  l_size=1, visibility=None):
    """
    Plot all 2D joints on the image
    :param image
    :param data: dict with 2D and 3D info
    :param p_size: size of the points
    :param l_size: size of the limbs
    """
    img = image.copy()
    if img.dtype == np.float32:
        img = clip_to_max(img, max=1.0)
        img *= 255
    else:
        img = clip_to_max(img, max=255)

    ubyte_img = img.astype(np.uint8)
    img = cv2.cvtColor(ubyte_img,
                       cv2.COLOR_BGR2RGB)

    if visibility is None:
        visibility = [True] * pose_2d.shape[0]

    for lid, (p0, p1) in enumerate(utils.LIMBS_3D):
        x0, y0 = pose_2d[p0].astype(np.int)
        x1, y1 = pose_2d[p1].astype(np.int)

        if visibility[p0]:
            cv2.circle(img, (x0, y0), p_size, _COLORS[_C_LIMBS_IDX[lid]], -1)

        if visibility[p1]:
            cv2.circle(img, (x1, y1), p_size, _COLORS[_C_LIMBS_IDX[lid]], -1)

        if visibility[p0] and visibility[p1]:
            cv2.line(img, (x0, y0), (x1, y1), _COLORS[_C_LIMBS_IDX[lid]], l_size, 16)

    return img


def clip_to_max(image, max):
    """
    Force the image to have maximum value 1
    :param image
    :return: image with max 1
    """
    shape = image.shape
    img = image.flatten()
    idx = np.where(img > max)[0]
    img[idx] = max

    new_img = img.reshape(shape)
    return new_img


def plot_pose(pose, c=1, r=1, o=1, color=None, dark=False,
              fig=None, planes=False, img=True):
    """Plot the 3D pose showing the joint connections."""
    from mpl_toolkits.mplot3d import Axes3D

    if dark:
        plt.style.use('dark_background')

    assert (pose.ndim == 2)
    if pose.shape[0] != 3:
        pose = pose.transpose([1, 0])

    if fig is None:
        fig = plt.figure(num=None, figsize=(8, 8),
                         dpi=100, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(r, c, o, projection='3d')
    for lid, (p0, p1) in enumerate(utils.LIMBS_3D):
        if color is None:
            col = '#{:02x}{:02x}{:02x}'.format(*_COLORS[_C_LIMBS_IDX[lid]])
        else:
            col = color
        ax.plot([pose[0, p0], pose[0, p1]],
                [pose[1, p0], pose[1, p1]],
                [pose[2, p0], pose[2, p1]], c=col)
        ax.scatter(pose[0, p0], pose[1, p0], pose[2, p0], c=col,
                   marker='o', edgecolor=col)
        ax.scatter(pose[0, p1], pose[1, p1], pose[2, p1], c=col,
                   marker='o', edgecolor=col)

    val = np.max([np.abs(pose.min()), np.abs(pose.max())])
    smallest = - val
    largest = val
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)
    ax.set_aspect('equal')

    if not planes:
        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

    fig.canvas.draw()

    if img:
        w, h = fig.canvas.get_width_height()
        image = np.fromstring(fig.canvas.tostring_rgb(),
                              dtype='uint8')
        image = image.reshape([w, h, 3])
        return image

    plt.show()
    return fig


def plot_poses(pose, gt, c=1, r=1, o=1, dark=False,
               planes=True, img=True, fig=None, angle=None):
    """Plot the 3D poses showing the joint connections."""
    from mpl_toolkits.mplot3d import Axes3D

    if dark:
        plt.style.use('dark_background')

    assert (pose.ndim == 2)
    assert (gt.ndim == 2)

    if pose.shape[0] != 3:
        pose = pose.transpose([1, 0])

    if gt.shape[0] != 3:
        gt = gt.transpose([1, 0])

    assert (pose.shape == gt.shape)

    if fig is None:
        fig = plt.figure(num=None, figsize=(8, 8),
                         dpi=100, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(r, c, o, projection='3d')
    col_preds = '#ff0000'
    col_gt = '#0000ff'

    for lid, (p0, p1) in enumerate(utils.LIMBS_3D):
        ax.plot([gt[0, p0], gt[0, p1]],
                [gt[1, p0], gt[1, p1]],
                [gt[2, p0], gt[2, p1]], c=col_gt)
        ax.scatter(gt[0, p0], gt[1, p0], gt[2, p0], c=col_gt,
                   marker='o', edgecolor=col_gt)
        ax.scatter(gt[0, p1], gt[1, p1], gt[2, p1], c=col_gt,
                   marker='o', edgecolor=col_gt)

        ax.plot([pose[0, p0], pose[0, p1]],
                [pose[1, p0], pose[1, p1]],
                [pose[2, p0], pose[2, p1]], c=col_preds)
        ax.scatter(pose[0, p0], pose[1, p0], pose[2, p0], c=col_preds,
                   marker='o', edgecolor=col_preds)
        ax.scatter(pose[0, p1], pose[1, p1], pose[2, p1], c=col_preds,
                   marker='o', edgecolor=col_preds)

    val = np.max([np.abs(pose.min()), np.abs(pose.max()),
                  np.abs(gt.min()), np.abs(gt.max())])
    smallest = - val
    largest = val
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)
    ax.set_aspect('equal')

    if not planes:
        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

    if angle:
        ax.view_init(30, angle)

    fig.canvas.draw()

    if img:
        w, h = fig.canvas.get_width_height()
        image = np.fromstring(fig.canvas.tostring_rgb(),
                              dtype='uint8')
        image = image.reshape([w, h, 3])
        plt.close('all')
        return image

    return fig
