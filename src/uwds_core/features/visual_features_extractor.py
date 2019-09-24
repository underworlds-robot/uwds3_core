import cv2
import keras
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2

class VisualFeaturesExtractor(object):
    """
    """
    def __init__(self, model_type="MobileNet", weights="imagenet", input_shape=(128,128, 3)):
        """
        """
        model_types = ["MobileNet", "MobileNetV2"]
        if model_type not in model_types:
            raise ValueError("Invalid model type")
        assert input_shape[0] > 32
        assert input_shape[1] > 32

        self.input_shape = input_shape

        if model_type == "MobileNet":
            self.model = MobileNet(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)
        if model_type == "MobileNetV2":
            self.model = MobileNetV2(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)

    def extract(self, frame, detection=None):
        """
        """
        if detection is not None:
            x = detection[0]
            y = detection[1]
            w = detection[2]
            h = detection[3]
            crop_frame = frame[y:y+h, x:x+w]
        else:
            crop_frame = frame
        frame_resized = cv2.resize(crop_frame, (self.input_shape[0], self.input_shape[1]))
        temp = frame_resized[0]
        frame_resized[0] = frame_resized[3]
        frame_resized[3] = temp
        x = image.img_to_array(frame_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)[0]
        return features
