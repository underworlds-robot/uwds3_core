import cv2
import numpy as np
from uwds3_core.utils.transformations import *
from .facial_landmarks_estimator import RIGHT_EYE_CORNER, LEFT_EYE_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER, CHIN, NOSE


class HeadPoseEstimator(object):
    def __init__(self):
        self.model_3d = np.float32([[0.0, 0.0, 0.0], # nose
                                    [0.0, -330.0, -65.0], # chin
                                    [-225.0, 170.0, -135.0], # left eye corner
                                    [225.0, 170.0, -135.0], # right eye corner
                                    [-150.0, -150.0, -125.0], # left mouth corner
                                    [150.0, -150.0, -125.0]]) / 1000/4.5 # right mouth corner

    def estimate(self, shape, camera_matrix, dist_coeffs, previous_head_pose=None):

        points_2d = np.float32([[shape[NOSE][0], shape[NOSE][1]],
                                [shape[CHIN][0], shape[CHIN][1]],
                                [shape[LEFT_EYE_CORNER][0], shape[LEFT_EYE_CORNER][1]],
                                [shape[RIGHT_EYE_CORNER][0], shape[RIGHT_EYE_CORNER][1]],
                                [shape[LEFT_MOUTH_CORNER][0], shape[LEFT_MOUTH_CORNER][1]],
                                [shape[RIGHT_MOUTH_CORNER][0], shape[RIGHT_MOUTH_CORNER][1]]])

        if previous_head_pose is None:
            _, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            r, t = previous_head_pose
            _, rot, trans = cv2.solvePnP(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=r, tvec=t)
        if trans[2] > 0:
            success = False
        else:
            success = True
        return success, rot, trans
