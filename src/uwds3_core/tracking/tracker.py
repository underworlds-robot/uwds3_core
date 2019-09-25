import numpy as np
from .track import Track
from .linear_assignment import LinearAssignment, iou_distance, cosine_distance


class Tracker(object):
    def __init__(self,
                 metric,
                 min_distance=0.5,
                 n_init=6,
                 max_disappeared=5,
                 max_age=8):

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.tracks = []
        self.iou_assignment = LinearAssignment(metric, min_distance=min_distance)

    def update(self, rgb_image, detections):
        matches, unmatched_detections, unmatched_tracks = self.iou_assignment.match(self.tracks, detections)

        for detection_indice, track_indice in matches:
            self.tracks[track_indice].update(rgb_image, detections[detection_indice])

        for track_indice in unmatched_tracks:
            if self.tracks[track_indice].is_confirmed():
                self.tracks[track_indice].predict(rgb_image)
            else:
                self.tracks[track_indice].mark_missed()

        for detection_indice in unmatched_detections:
            self.start_track(rgb_image, detections[detection_indice])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks

    def start_track(self, rgb_image, detection):
        if detection.class_label == "person":
            self.tracks.append(Track(rgb_image,
                                     detection,
                                     self.n_init,
                                     self.max_disappeared,
                                     self.max_age,
                                     use_correlation_tracker=False))
        else:
            self.tracks.append(Track(rgb_image,
                                     detection,
                                     self.n_init,
                                     self.max_disappeared,
                                     self.max_age,
                                     use_correlation_tracker=True))
        return len(self.tracks)-1
