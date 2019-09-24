import numpy as np
import cv2
import dlib
import yaml
from detection import Detection

class OpenCVDNNDetector(object):
    """  """

    def __init__(self, model, weights, config_file_path, input_size, max_overlap_ratio=0.8, detector_confidence=1.0, swapRB=False):
        """  """
        with open(config_file_path, "r") as f:
            self.config = yaml.load(f)
        self.model = cv2.dnn.readNetFromTensorflow(model, weights)
        self.input_size = input_size
        self.detector_confidence = detector_confidence
        self.max_overlap_ratio = max_overlap_ratio
        self.swapRB = swapRB

    def detect(self, frame):
        """
        """
        frame_resized = cv2.resize(frame, (self.input_size, self.input_size))

        self.model.setInput(cv2.dnn.blobFromImage(frame_resized, swapRB=self.swapRB))

        detections = self.model.forward()
        filtered_detections = []

        detection_per_class = {}
        score_per_class = {}

        rows = frame_resized.shape[0]
        cols = frame_resized.shape[1]

        height_factor = frame.shape[0]/float(self.input_size)
        width_factor = frame.shape[1]/float(self.input_size)

        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]
            if class_id in self.config:
                if self.config[class_id]["activated"] is True:
                    if confidence > self.config[class_id]["confidence_threshold"]:

                        class_label = self.config[class_id]["label"]
                        x_top_left = int(detections[0, 0, i, 3] * cols)
                        y_top_left = int(detections[0, 0, i, 4] * rows)
                        x_right_bottom = int(detections[0, 0, i, 5] * cols)
                        y_right_bottom = int(detections[0, 0, i, 6] * rows)

                        x_top_left = int(width_factor * x_top_left)
                        y_top_left = int(height_factor * y_top_left)
                        x_right_bottom = int(width_factor * x_right_bottom)
                        y_right_bottom = int(height_factor * y_right_bottom)

                        x_top_left = 0 if x_top_left < 0 else x_top_left
                        y_top_left = 0 if y_top_left < 0 else y_top_left
                        x_right_bottom = frame.shape[1]-1 if x_right_bottom > frame.shape[1]-1 else x_right_bottom
                        y_right_bottom = frame.shape[0]-1 if y_right_bottom > frame.shape[0]-1 else y_right_bottom

                        bbox = [x_top_left, y_top_left, x_right_bottom, y_right_bottom, confidence*self.detector_confidence]
                        if class_label not in detection_per_class:
                            detection_per_class[class_label] = []
                        if class_label not in score_per_class:
                            score_per_class[class_label] = []
                        detection_per_class[class_label].append(bbox)

        for class_label, dets in detection_per_class.items():
            filtered_dets = self.non_max_suppression(np.array(dets), self.max_overlap_ratio)
            for d in filtered_dets:
                filtered_detections.append(Detection(d[0], d[1], d[2], d[3], d[4]*self.detector_confidence, class_label))

        return filtered_detections

    def non_max_suppression(self, boxes, max_bbox_overlap):

        if len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores is not None:
            idxs = np.argsort(scores)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > max_bbox_overlap)[0])))

        return boxes[pick]
