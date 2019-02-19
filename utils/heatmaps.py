# -*- coding: utf-8 -*-
"""
Created on Aug 03 18:50 2018

@author: Denis Tome'
"""
import cv2
import numpy as np
from scipy.stats import multivariate_normal
from utils.conversions import standardize_pose

__all__ = [
    'generate_heatmap',
    'pose_to_heatmaps',
    'overimpose_to_img',
    'grayscale_to_heatmap',
    'heatmaps_to_pose',
    'heatmaps_to_poses'
]


def generate_heatmap(size, point, sigma_x, sigma_y):
    """Generate single heatmap centered in at the
    specified position with a diagonal covariance matrix

    Arguments:
        size {int} -- square heatmap
        point {list} -- contains (u, v) joint position
        sigma_x {float} -- sigma value
        sigma_y {float} -- sigma value

    Returns:
        numpy array -- format (Size x Size)
    """

    try:
        cov_matrix = np.eye(2) * ([sigma_x ** 2, sigma_y ** 2])

        x, y = np.mgrid[0:size, 0:size]
        pos = np.dstack((x, y))
        rv = multivariate_normal([point[1],
                                  point[0]],
                                 cov_matrix)

        hm_pdf = rv.pdf(pos)
        hmap = np.multiply(hm_pdf,
                           np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov_matrix)))
        idx = np.where(hmap.flatten() <= np.exp(-4.6052))
        hmap.flatten()[idx] = 0
    except ZeroDivisionError:
        return np.zeros([size, size])

    return hmap


def pose_to_heatmaps(pose_2d, pose_3d, size, sigma):
    """Generate set of heatmaps for a pose

    Arguments:
        pose_2d {numpy array} -- format (N_JOINTS x 2)
        pose_3d {numpy array} -- format (N_JOINTS x 2) for checking projections
        size {int} -- size of a single square heatmap
        sigma {float} -- sigma value

    Returns:
        numpy array -- format (size x size x N_JOINTS)
    """

    pose_2d = standardize_pose(pose_2d)
    pose_3d = standardize_pose(pose_3d)

    assert np.min(pose_2d.shape) == 2
    assert np.min(pose_3d.shape) == 3

    heatmaps = np.zeros([size, size, pose_2d.shape[0]])
    for idx, joint in enumerate(pose_2d):
        if pose_3d[idx, -1] > 0:
            continue

        heatmaps[:, :, idx] = generate_heatmap(size,
                                               joint,
                                               sigma,
                                               sigma)

    return heatmaps


def overimpose_to_img(image, heatmap, img_w=0.6, hm_w=0.4):
    """Plot heatmap on top of an image

    Arguments:
        image {numpy array} -- RGB image matrix
        heatmap {numpy arrat} -- format (SIZE x SIZE x N_JOINTS)

    Keyword Arguments:
        img_w {float} -- alpha for the image (default: {0.6})
        hm_w {float} -- alpha for the heatmaps (default: {0.4})

    Returns:
        numpy array -- RGB image with the heatmaps
    """

    assert image.ndim == 3
    assert heatmap.ndim == 2
    assert image.shape[:2] == heatmap.shape
    assert image.shape[-1] == 3

    hm = cv2.normalize(heatmap, dst=None, alpha=0, beta=255,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img = cv2.normalize(image, dst=None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    img_3ch = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    des_img = cv2.addWeighted(img, img_w,
                              img_3ch, hm_w, 0)
    return des_img


def grayscale_to_heatmap(image):
    """Color version of a grayscale heatmap

    Arguments:
        image {numpy array} -- gray scale, format (SIZE x SIZE)

    Returns:
        numpy array -- format (SIZE x SIZE x 3) RGB image
    """

    img = cv2.normalize(image, dst=None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img_3ch = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    return img_3ch


def heatmaps_to_pose(heatmaps):
    """Heatmaps to 2D pose

    Arguments:
        heatmaps {numpy array} -- format (SIZE x SIZE x N_JOINTS)

    Returns:
        numpy array -- format (N_JOINTS x 2)
    """

    assert heatmaps.shape[1] == heatmaps.shape[-1]
    assert heatmaps.ndims == 3

    pose = np.zeros([heatmaps.shape[0], 2])
    size = heatmaps.shape[1:]
    for jid, hm in enumerate(heatmaps):
        y, x = np.unravel_index(np.argmax(hm.reshape(-1)), size)
        pose[jid] = [x, y]

    return pose


def heatmaps_to_poses(heatmaps):
    """Multiple heatmaps to mutliple 2D poses

    Arguments:
        heatmaps {numpy array} -- format (N_IMAGES x SIZE x SIZE x N_JOINTS)

    Returns:
        numpy array -- format (N_IMAGES x N_JOINTS x 2)
    """

    poses = np.zeros([heatmaps.shape[0],
                      heatmaps.shape[1],
                      2])
    for pid, p_hm in enumerate(heatmaps):
        poses[pid] = heatmaps_to_pose(p_hm)

    return poses
