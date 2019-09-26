import dlib
import numpy as np
import uuid
from tf.transformations import quaternion_matrix, translation_matrix


class HumanTrack(object):
    def __init__(self, person_track):
        self.uuid = str(uuid.uuid4())
        self.is_facing = False
        self.class_label = "person"
        self.bbox = person_track.bbox
        self.body_part_types = ["person", "face", "right_hand", "left_hand"]
        self.body_part_tracks = {}
        self.body_part_tracks["person"] = person_track

    def update_track(self, track_label, track):
        if track_label == "person":
            self.bbox = track.bbox
        self.body_part_tracks[track_label] = track

    def mark_missed(self, track_label=None):
        if track_label is None:
            self.body_part_tracks["person"].mark_missed()
        else:
            if self.body_part_tracks[track_label].is_deleted():
                del self.body_part_tracks[track_label]

    def is_facing(self):
        if "face" in self.body_part_tracks:
            return self.body_part_tracks["face"].is_confirmed()
        else:
            return False

    def get_head_pose(self):
        if "face" in self.body_part_tracks:
            return (self.body_part_tracks["face"].rotation, self.body_part_tracks["face"].translation)
        else:
            return (None, None)

    def is_perceived(self):
        return self.body_part_tracks["person"].is_perceived()

    def is_confirmed(self):
        return self.body_part_tracks["person"].is_confirmed()

    def is_deleted(self):
        return self.body_part_tracks["person"].is_deleted()
