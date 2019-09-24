import dlib
import numpy as np
from tf.transformations import quaternion_matrix, translation_matrix


class Entity(object):
    def __init__(self, parent_id, track, t, q):
        self.uuid = track.uuid
        self.label = track.label
        self.parent = parent_id
        self.transform = np.dot(quaternion_matrix(q), translation_matrix(t))
        self.shape = Shape()

    def update(self, track):
        pass

    def predict(self):
        pass
