import rospy
import numpy as np
import sensor_msgs
import tf2_ros
import math
import cv2
import message_filters
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from .utils.transformations import *
from .detection.opencv_dnn_detector import OpenCVDNNDetector
from .detection.hog_face_detector import HOGFaceDetector
from .storage.internal_simulator import InternalSimulator
from .estimation.dense_optical_flow_estimator import DenseOpticalFlowEstimator
from .estimation.facial_landmarks_estimator import FacialLandmarksEstimator, NOSE, POINT_OF_SIGHT
from .estimation.head_pose_estimator import HeadPoseEstimator
from .features.visual_features_extractor import VisualFeaturesExtractor
from .tracking.tracker import Tracker
from .tracking.linear_assignment import iou_distance

def transformation_matrix(t, q):
    translation_mat = translation_matrix(t)
    rotation_mat = quaternion_matrix(q)
    return np.dot(translation_mat, rotation_mat)



class UnderworldsCore(object):
    def __init__(self):

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "map")

        self.use_gui = rospy.get_param("~use_gui", True)

        rospy.loginfo("Subscribing to /{} topic...".format(self.camera_info_topic))
        self.camera_info = None
        self.camera_frame_id = None
        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic, sensor_msgs.msg.CameraInfo, self.camera_info_callback)

        self.detector_model_filename = rospy.get_param("~detector_model_filename", "")
        self.detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        self.detector_config_filename = rospy.get_param("~detector_config_filename", "")
        self.detector = OpenCVDNNDetector(self.detector_model_filename,
                                          self.detector_weights_filename,
                                          self.detector_config_filename,
                                          300)

        self.face_detector = HOGFaceDetector()

        self.visual_features_extractor = VisualFeaturesExtractor("MobileNetV2", weights="imagenet")

        self.internal_simulator = InternalSimulator()

        self.shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        self.landmark_estimator = FacialLandmarksEstimator(self.shape_predictor_config_filename)

        self.head_pose_estimator = HeadPoseEstimator()

        self.optical_flow_estimator = DenseOpticalFlowEstimator()

        self.flow = None

        self.bridge = CvBridge()

        self.tracker = Tracker(iou_distance, min_distance=0.7)

        self.use_depth = rospy.get_param("~use_depth", False)

        self.n_frame = rospy.get_param("~n_frame", 2)
        self.frame_count = 0

        self.only_faces = rospy.get_param("~only_faces", True)

        self.visualization_publisher = rospy.Publisher("uwds3_core/visualization_image", sensor_msgs.msg.Image, queue_size=1)

        if self.use_depth is True:
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, sensor_msgs.msg.Image)
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, sensor_msgs.msg.Image)

            self.sync = message_filters.TimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], 10)
            self.sync.registerCallback(self.observation_callback_with_depth)
        else:
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, sensor_msgs.msg.Image, self.observation_callback, queue_size=1)


        #self.depth_estimator = DepthEstimator()

        #self.sync = message_filters.TimeSynchronizer([self.tracks_subscriber, self.rgb_image_subscriber, self.depth_image_subscriber], 10)
        #self.sync.registerCallback(self.observation_callback)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def get_last_transform_from_tf2(self, source_frame, target_frame):
        try:
            trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            return True, [x, y, z], [rx, ry, rz, rw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

    def observation_callback(self, rgb_image_msg):
        if self.camera_info is not None:

            bgr_image = self.bridge.imgmsg_to_cv2(rgb_image_msg)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            viz_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            timer1 = cv2.getTickCount()

            detections = []
            if self.frame_count % self.n_frame == 0:
                detections = self.face_detector.detect(rgb_image)
            if self.frame_count % self.n_frame == 1:
                    detections += self.detector.detect(rgb_image)
            self.frame_count += 1
            tracks = self.tracker.update(rgb_image, detections)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer1)
            detection_fps = "Detection and track fps : %0.4fhz" % fps
            #print(detection_fps)
            cv2.putText(viz_frame, detection_fps, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for track in tracks:
                if track.is_confirmed():
                    tl_corner = (int(track.bbox.left()), int(track.bbox.top()))
                    br_corner = (int(track.bbox.right()), int(track.bbox.bottom()))
                    if track.class_label == "face":
                        shape = self.landmark_estimator.estimate(rgb_image, (int(track.bbox.left()), int(track.bbox.top()), int(track.bbox.right()), int(track.bbox.bottom())))
                        for (x, y) in shape:
                            cv2.circle(viz_frame, (x, y), 1, (0, 255, 0), -1)
                        success, rot, trans = self.head_pose_estimator.estimate(shape, self.camera_matrix, self.dist_coeffs)
                        #offset = euler_matrix(math.radians(0), math.radians(0), math.radians(0), "rxyz")
                        #head_pose_rot = inverse_matrix(euler_matrix(rot[0], rot[1], rot[2], 'rxyz'))
                        #new_rot = euler_from_matrix(head_pose_rot, "rxyz")
                        #head_pose_offset = np.dot(head_pose_transform, offset)
                        #_, _, new_rot, new_trans, _ = decompose_matrix(head_pose_transform)
                        cv2.drawFrameAxes(viz_frame, self.camera_matrix, self.dist_coeffs, np.array(rot).reshape((3,1)), np.array(trans).reshape(3,1), 30)
                        _, t, q = self.get_last_transform_from_tf2(self.global_frame_id, self.camera_frame_id)
                        view_matrix = transformation_matrix(t, q)
                        head_pose_transform = compose_matrix(angles=[rot[0], rot[1], rot[2]], translate=[trans[0], trans[1], trans[2]])
                        head_pose_world = np.dot(view_matrix, head_pose_transform)
                        _, _, r, t, _ = decompose_matrix(head_pose_world)
                        q = quaternion_from_euler(r[0], r[1], r[2], "rxyz")
                        if track.uuid not in self.internal_simulator:
                            rospy.loginfo("load_urdf")
                            self.internal_simulator.load_urdf(track.uuid, "face.urdf", t, q)
                        else:
                            rospy.loginfo("update_entity")
                            self.internal_simulator.update_entity(track.uuid, t, q)

                        #self.internal_simulator.load_urdf(t.uuid, "face.urdf", t, q)
                            # success, t, q = self.get_last_transform_from_tf2(self.global_frame_id, self.camera_frame_id)
                            # if success:
                            #     q = quaternion_from_matrix(rot)

                    cv2.putText(viz_frame, track.uuid[:6], (tl_corner[0]+5, tl_corner[1]+25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
                    cv2.putText(viz_frame, track.class_label, (tl_corner[0]+5, tl_corner[1]+45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
                    cv2.rectangle(viz_frame, tl_corner, br_corner, (255, 255, 0), 2)
            viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
            self.visualization_publisher.publish(viz_img_msg)