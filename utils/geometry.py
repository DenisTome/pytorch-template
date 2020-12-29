# -*- coding: utf-8 -*-
"""
Geometry utilities

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""

__version__ = "0.1.0"
__all__ = [
    'get_vertex_id_inside_quad',
    'get_plane_from_bounding_box',
    'get_tangent_vector_from_points'
]

import numpy as np
import open3d as o3d


def get_vertex_id_inside_quad(points, area) -> np.array:
    """Get vertex ids for those vertices that are inside the definde area

    Args:
        points (np.array): vertices
        area (np.array): quad corresponding to the area of interest

    Returns:
        np.array: list of indices
    """

    assert len(area) == 4, "Input is not a quad!"

    top_left = np.min(area, axis=0)
    bottom_right = np.max(area, axis=0)

    idx = np.where((points[:, 0] >= top_left[0]) & (points[:, 0] <= bottom_right[0]) & (
        points[:, 1] >= top_left[1]) & (points[:, 1] <= bottom_right[1]))[0]

    if len(idx) > 0:
        return idx

    return None


def get_plane_from_bounding_box(bb: o3d.geometry.OrientedBoundingBox,
                                z_axis: int = 2) -> np.array:
    """Generate plane from bounding box

    Args:
        bb (OrientedBoundingBox): bounding box
        z_axis (int, optional): z axis index. Defaults to 2.

    Returns:
        np.array: vertices
    """

    bb_vertices = np.asarray(bb.get_box_points())

    # remove depth
    avg_depth = bb_vertices[:, z_axis].mean()
    bb_vertices[:, z_axis] = avg_depth

    # remove duplicate vertices due to squeezing them on the z axis
    copy_box = np.around(bb_vertices, decimals=10)
    _, indices = np.unique(copy_box, axis=0, return_index=True)

    return bb_vertices[indices]


def get_tangent_vector_from_points(point_a: np.array,
                                   point_b: np.array,
                                   norm: bool = True) -> np.array:
    """Compute tangent vector from the input points

    Args:
        point_a (np.array): point A
        point_b (np.array): point B

    Returns:
        np.array: tangent vector
    """

    vector = point_a - point_b
    if norm:
        norm = np.linalg.norm(vector, 2)
        return vector / norm

    return vector
