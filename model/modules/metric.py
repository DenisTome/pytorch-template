# -*- coding: utf-8 -*-
"""
Custom metric inheriting from base_metric

@author: Denis Tome'

Copyright Epic Games, Inc. All Rights Reserved.

"""
import numpy as np
from base import BaseMetric
from utils import compute_3d_joint_error


class PoseError(BaseMetric):
    """
    Average Euclidean distance between the gt
    and the predicted 3D poses.
    """

    def evaluate(self, pred: np.array, gt: np.array) -> float:
        """Compute metric

        Args:
            pred (np.array): predictions
            gt (np.array): ground truth

        Returns:
            float: result of evaluation
        """

        overall_err = 0.0
        for pose_in, pose_target in zip(pred, gt):
            error = compute_3d_joint_error(pose_in,
                                           pose_target)
            overall_err += np.mean(error, axis=0)

        return overall_err / pred.shape[0]

    @property
    def desc(self):
        """Get description of the metric"""
        return 'MeanJointError'
