import dlib
import uuid
import numpy as np


class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED = 3
    DELETED = 4


class Track(object):
    def __init__(self,
                 rgb_image,
                 detection,
                 n_init,
                 max_disappeared,
                 max_age,
                 use_correlation_tracker=True,
                 feature=None):

        self.uuid = str(uuid.uuid4())

        self.bbox = dlib.rectangle(int(detection.bbox.left()),
                                   int(detection.bbox.top()),
                                   int(detection.bbox.right()),
                                   int(detection.bbox.bottom()))

        self.class_label = detection.class_label

        self.k_past = dlib.rectangles()

        if use_correlation_tracker is True:
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(rgb_image, self.bbox)
        else:
            self.tracker = None

        self.state = TrackState.TENTATIVE

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age

        self.translation = None
        self.rotation = None

        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None

        self.missed = 0
        self.age = 1
        self.hits = 1
        self.since_update = 1

    def update(self, rgb_image, detection):
        if self.tracker is not None:
            self.tracker.update(rgb_image, detection.bbox)
            self.bbox = self.tracker.get_position()
        else:
            self.bbox = detection.bbox
        self.class_label = detection.class_label
        self.hits += 1
        self.age = 0
        self.since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        if self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def predict(self, rgb_image):
        if self.age > self.max_age:
            self.mark_missed()
        else:
            if self.tracker is not None:
                self.tracker.update(rgb_image)
                self.bbox = self.tracker.get_position()
            else:
                pass
            self.age += 1
            self.since_update = 0

    def mark_missed(self):
        self.since_update += 1
        if self.state == TrackState.TENTATIVE:
            if self.since_update > self.max_disappeared:
                self.state = TrackState.DELETED
        if self.state == TrackState.CONFIRMED:
            if self.since_update > self.max_disappeared:
                self.state = TrackState.OCCLUDED
        elif self.state == TrackState.OCCLUDED:
            if self.since_update > self.max_age:
                self.state = TrackState.DELETED

    def is_perceived(self):
        if not self.is_deleted():
            return self.state != TrackState.OCCLUDED
        else:
            return False

    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED or self.state == TrackState.OCCLUDED

    def is_deleted(self):
        return self.state == TrackState.DELETED

    def __str__(self):
        return "rect : {} for class : {}".format(self.bbox, self.class_label)
