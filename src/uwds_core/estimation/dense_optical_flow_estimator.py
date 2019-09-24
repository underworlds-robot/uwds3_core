import cv2


class DenseOpticalFlowEstimator(object):
    def __init__(self):
        self.previous_frame = None

    def estimate(self, frame):
        if first_frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.previous_frame, gray, None, 0.5, 1, 20, 1, 5, 1.2, 0)
        self.previous_frame = gray
        return flow
