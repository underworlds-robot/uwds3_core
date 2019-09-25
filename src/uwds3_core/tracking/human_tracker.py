import numpy as np
from .linear_assignment import LinearAssignment, iou_distance, euler_distance
from uwds3_core.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_core.estimation.head_pose_estimator import HeadPoseEstimator
from .tracker import Tracker
from .human_track import HumanTrack


class HumanTracker(Tracker):
    def __init__(self,
                 landmarks_prediction_model_filename,
                 min_distance=0.5,
                 n_init=6,
                 max_disappeared=5,
                 max_age=8):

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.tracks = []
        self.iou_assignment = LinearAssignment(iou_distance, min_distance=min_distance)
        self.human_tracks = []
        self.euler_assignment = LinearAssignment(euler_distance)

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

        face_tracks = [t for t in self.tracks if t.class_label=="face"]
        person_tracks = [t for t in self.tracks if t.class_label=="person"]

        # matches, unmatched_persons, unmatched_humans = self.iou_assignment.match(self.human_tracks, person_tracks)
        #
        # for human_indice, person_indice in matches:
        #     self.human_tracks[human_indice].update_track("person", self.tracks[person_indice])
        #
        # for human_indice in unmatched_humans:
        #     self.human_tracks[human_indice].mark_missed()
        #
        # for person_indice in unmatched_persons:
        #     self.start_human_track(self.tracks[person_indice])
        #
        # matches, unmatched_faces, unmatched_person = self.euler_assignment.match(self.human_tracks, face_tracks)
        #
        # for human_indice, face_indice in matches:
        #     self.human_tracks[human_indice].update_track("face", self.tracks[face_indice])

        return self.human_tracks

    def start_human_track(self, person_track):
        self.human_tracks.append(HumanTrack(person_track))
        return len(self.human_tracks)-1
