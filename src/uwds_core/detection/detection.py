import dlib
import numpy as np


class Detection(object):
    def __init__(self,
                 x_top_left,
                 y_top_left,
                 x_right_bottom,
                 y_right_bottom,
                 confidence,
                 class_label,
                 feature=None):

        self.bbox = dlib.rectangle(int(x_top_left),
                                   int(y_top_left),
                                   int(x_right_bottom),
                                   int(y_right_bottom))
        self.confidence = confidence
        self.class_label = class_label

        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None

    def __str__():
        return "rect : {} for class : {} with {} confidence".format(self.bbox, self.class_label, self.confidence)
