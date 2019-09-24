import cv2
import numpy as np

class HistogramFeaturesExtractor(object):
    def __init__(self):
        pass

    def extract(self, frame, detection=None):
        """ Extract hue channel histogram as a features vector """
        if detection is not None:
            x = detection[0]
            y = detection[1]
            w = detection[2]
            h = detection[3]
            crop_frame = frame[y:y+h, x:x+w]
        else:
            crop_frame = frame
        try:
            frame_resized = cv2.resize(crop_frame, (128, 128))
        except:
            print detection
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist / np.max(hist)
        return hist[:, 0]
