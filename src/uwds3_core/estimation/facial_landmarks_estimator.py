import numpy as np
import dlib
import cv2

POINT_OF_SIGHT = 27
RIGHT_EYE_CORNER = 36
LEFT_EYE_CORNER = 45
NOSE = 30
MOUTH_UP = 51
MOUTH_DOWN = 57
MOUTH_UP = 51
RIGHT_MOUTH_CORNER = 48
LEFT_MOUTH_CORNER = 54
RIGHT_EAR = 0
LEFT_EAR = 16
CHIN = 8


class FacialLandmarksEstimator(object):
    def __init__(self, shape_predictor_config_file):
        self.predictor = dlib.shape_predictor(shape_predictor_config_file)

    def estimate(self, rgb_image, face_track):
        bbox = face_track.bbox
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        shape = self.predictor(gray, dlib.rectangle(int(bbox.left()), int(bbox.top()), int(bbox.right()), int(bbox.bottom())))
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (int(shape.part(i).x), int(shape.part(i).y))
        return coords
