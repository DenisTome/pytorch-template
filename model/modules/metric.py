# -*- coding: utf-8 -*-
"""
Custom metric

@author: Denis Tome'

"""
import numpy as np
from base import BaseMetric
from utils import compute_3d_joint_error


class PoseError(BaseMetric):
    """
    Average Euclidean distance between the gt
    and the predicted 3D poses.
    """

    def eval(self, pred, gt):
        """Evaluate

        Arguments:
            pred {numpy array} -- prediction
            gt {numpy arrat} -- ground truth

        Returns:
            float -- error
        """

        overall_err = 0.0
        for pose_in, pose_target in zip(pred, gt):
            error = compute_3d_joint_error(pose_in,
                                           pose_target)
            overall_err += np.mean(error, axis=0)

        return overall_err / pred.shape[0]

    def _desc(self):
        return 'MeanJointError'

