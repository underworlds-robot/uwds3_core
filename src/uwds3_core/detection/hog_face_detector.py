import dlib
import numpy as np
from.detection import Detection
import cv2

class HOGFaceDetector(object):
    def __init__(self, detector_confidence=0.85):
        self.detector = dlib.get_frontal_face_detector()
        self.detector_confidence = detector_confidence

    def detect(self, rgb_image):
        height, width, _ = rgb_image.shape
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 0)
        return [self.rect_to_detection(r) for r in rects]

    def rect_to_detection(self, rect):
        return Detection(rect.left(), rect.top(), rect.right(), rect.bottom(), self.detector_confidence, "face")
