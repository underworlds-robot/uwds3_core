import numpy as np
from .linear_assignment import LinearAssignment, iou_distance, euler_distance
from uwds3_core.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_core.estimation.head_pose_estimator import HeadPoseEstimator
from .tracker import Tracker
from track import Track
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
        self.human_tracks = []
        self.iou_assignment = LinearAssignment(iou_distance, min_distance=min_distance)
        self.euler_assignment = LinearAssignment(euler_distance)
        self.facial_landmarks_estimator = FacialLandmarksEstimator(landmarks_prediction_model_filename)
        self.head_pose_estimator = HeadPoseEstimator()

    def update(self, rgb_image, detections, camera_matrix, dist_coeffs):
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

        for face_track in face_tracks:
            shape = self.facial_landmarks_estimator.estimate(rgb_image, face_track)
            if face_track.rotation is None or face_track.translation is None:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs)
            else:
                success, rot, trans = self.head_pose_estimator.estimate(shape, camera_matrix, dist_coeffs, previous_head_pose=(face_track.rotation.reshape((3,1)), face_track.translation.reshape((3,1))))
            if success is True:
                face_track.rotation = rot.reshape((3,))
                face_track.translation = trans.reshape((3,))

        person_tracks = [t for t in self.tracks if t.class_label=="person"]

        matches, unmatched_persons, unmatched_humans = self.iou_assignment.match(self.human_tracks, person_tracks)

        for person_indice, human_indice in matches:
            self.human_tracks[human_indice].update_track("person", self.tracks[person_indice])

        for human_indice in unmatched_humans:
            self.human_tracks[human_indice].mark_missed()

        for person_indice in unmatched_persons:
            self.start_human_track(self.tracks[person_indice])

        matches, unmatched_faces, unmatched_humans = self.euler_assignment.match(self.human_tracks, face_tracks)

        for face_indice, human_indice  in matches:
            self.human_tracks[human_indice].update_track("face", self.tracks[face_indice])

        self.human_tracks = [t for t in self.human_tracks if not t.is_deleted()]

        return self.human_tracks

    def start_human_track(self, person_track):
        self.human_tracks.append(HumanTrack(person_track))
        return len(self.human_tracks)-1

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
