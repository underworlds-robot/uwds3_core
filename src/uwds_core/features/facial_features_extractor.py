import cv2
import numpy as np
import dlib

class FacialFeaturesExtractor(object):
    def __init__(self, model_file_path, input_size, swapRB=True):
        self.model = cv2.dnn.readNetFromTorch(model_file_path)
        self.swapRB = swapRB

    def extract(rgb_frame: Type[np.array], bbox : Type[dlib.rectangle]):
        crop_frame = frame[bbox.left():bbox.right(), bbox.top():bbox.bottom()]
        frame_resized = cv2.resize(crop_frame, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=self.swapRB, crop=False)
        self.model.setInput(blob)
        if w < 20 or h < 20:
			return False, None
        vec = self.model.forward()
        return True, vec.flatten()
