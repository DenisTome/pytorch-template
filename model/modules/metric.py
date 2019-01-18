# -*- coding: utf-8 -*-
"""
Created on Jan 18 17:32 2019

@author: Denis Tome'
"""
import numpy as np
from base.base_metric import BaseMetric
from utils import compute_3d_joint_error


class AvgPoseError(BaseMetric):
    """
    Average Euclidean distance between the gt
    and the predicted 3D poses.
    """

    def eval(self, pred, gt):
        """
        :param **kwargs:
        :param pred: 3D pose given as input
        :param gt: 3D gt pose
        :return: average 3D error of the pose
        """
        overall_err = 0.0

        pid = 0
        for pose_in, pose_target in zip(pred, gt):
            error = compute_3d_joint_error(pose_in,
                                           pose_target)
            overall_err += np.mean(error, axis=0)
            pid += 1

        return overall_err / pred.shape[0]

    def _desc(self):
        return 'MeanJointError'
