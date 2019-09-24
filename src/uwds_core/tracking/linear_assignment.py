import dlib
import numpy as np
import math
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.spatial.distance import cosine


class LinearAssignment(object):

    def __init__(self, cost_metric, min_distance):
        self.cost_metric = cost_metric
        self.min_distance = min_distance

    def match(self, tracks, detections):

        if(len(tracks) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # Create the cost matrix
        C = np.zeros((len(detections), len(tracks)), dtype=np.float32)

        # Compute the cost matrix
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                C[d, t] = self.cost_metric(det, trk)

        # Run the optimization problem
        M = linear_assignment(C)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in M[:, 0]):
                unmatched_detections.append(d)
        unmatched_tracks = []
        for t, trk in enumerate(tracks):
            if(t not in M[:, 1]):
                unmatched_tracks.append(t)

        matches = []
        for m in M:
            if(C[m[0], m[1]] > self.min_distance):
                unmatched_detections.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


def iou_distance(track, detection):
    x_top_left = min(track.bbox.left(), detection.bbox.left())
    y_top_left = min(track.bbox.top(), detection.bbox.top())
    x_right_bottom = max(track.bbox.right(), detection.bbox.right())
    y_right_bottom = max(track.bbox.bottom(), detection.bbox.bottom())
    intersection_area = dlib.rectangle(int(x_top_left), int(y_top_left), int(x_right_bottom), int(y_right_bottom)).area()
    track_area = track.bbox.area()
    detection_area = detection.bbox.area()
    return intersection_area / float(track_area + detection_area)


def cosine_distance(track, detection):
    return cosine(track.feature, detection.feature)


def euler_distance(track, detection):
    t_cx, t_cy = track.bbox.center()
    d_cx, d_cy = detection.bbox.center()
    return math.sqrt(pow(t_cx-d_cx, 2)+pow(t_cy-d_cy, 2))
