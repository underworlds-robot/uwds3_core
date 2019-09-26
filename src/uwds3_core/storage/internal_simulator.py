import rospy
import numpy as np
import pybullet as p
import pybullet_data
import time
import tf2_ros
import geometry_msgs
import sensor_msgs
from uwds3_core.utils.transformations import *
from tf2_ros import Buffer, TransformListener
from pybullet import JOINT_FIXED

from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge
import colorsys


class HumanVisualModel(object):
    FOV = 60.0 # human field of view
    WIDTH = 10 # image width resolution for rendering
    HEIGHT = 10  # image height resolution for rendering
    CLIPNEAR = 0.3 # clipnear
    CLIPFAR = 1e+3 # clipfar
    ASPECT = 1.333 # aspect ratio for rendering
    SACCADE_THRESHOLD = 0.01 # angular variation in rad/s
    SACCADE_ESPILON = 0.005 # error in angular variation
    FOCUS_DISTANCE_FIXATION = 0.1 # focus distance when performing a fixation
    FOCUS_DISTANCE_SACCADE = 0.5 # focus distance when performing a saccade

class HumanGazeState(object):
    SACCADE = 0 #
    FIXATION = 1 #

class ObjectPhysicalState(object):
    INCONSISTENT = 0
    PLAUSIBLE = 1


class InternalSimulator(object):
    def __init__(self):

        rospy.loginfo("Subscribing to /tf topic...")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # set the parameters
        self.env_sdf_file_path = rospy.get_param("~env_sdf_file_path", "")
        self.robot_urdf_file_path = rospy.get_param("~robot_urdf_file_path", "r2d2.urdf")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "map")

        self.objects_cad_models_dir = rospy.get_param("~objects_cad_models_dir", "")
        self.robot_cad_models_dir = rospy.get_param("~robot_cad_models_dir", "")

        self.position_tolerance = rospy.get_param("~position_tolerance", 0.01)
        self.velocity_tolerance = rospy.get_param("~velocity_tolerance", 0.001)

        self.camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        self.time_step = rospy.get_param("~time_step", 1/240.0)

        # Initialize attributes
        self.simulator_entity_id_map = {}
        self.reverse_simulator_entity_id_map = {}

        self.simulator_joint_id_map = {}
        self.reverse_simulator_joint_id_map = {}

        self.entities_state = {}
        self.entities_last_update = {}

        self.camera_frame_id = None

        self.last_backup_id = None

        self.camera_info = None

        self.last_observation_time = None
        self.average_observation_duration = None

        self.bridge = CvBridge()
        self.visualization_publisher = rospy.Publisher("uwds3_core/debug_image", sensor_msgs.msg.Image, queue_size=1)

        # Setup the physics server
        use_gui = rospy.get_param("~use_gui", True)
        if use_gui is True:
            self.physics = p.connect(p.GUI)
        else:
            self.physics = p.connect(p.DIRECT)

        if p.isNumpyEnabled() is not True:
            rospy.logwarn("Numpy is not enabled in pybullet !")

        # Add search path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the ground
        self.simulator_entity_id_map[self.global_frame_id] = p.loadURDF("plane.urdf")

        # if self.robot_cad_models_dir != "":
        #     p.setAdditionalSearchPath(self.robot_cad_models_dir)

        if self.objects_cad_models_dir != "":
            p.setAdditionalSearchPath(self.objects_cad_models_dir)

        # Load the environment if any
        if self.env_sdf_file_path != "":
            self.simulator_entity_id_map["env"] = p.loadURDF(self.env_sdf_file_path)

        self.robot_loaded = False

        # Set the gravity
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # TODO
        #self.simulation_timer =

        # Subscribe to joint state message
        rospy.loginfo("Subscribing to /joint_states topic...")
        self.joint_state_subscriber = rospy.Subscriber("/joint_states", sensor_msgs.msg.JointState, self.joint_states_callback)


        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")
        rospy.loginfo("Subscribing to /{} topic...".format(self.camera_info_topic))
        self.camera_info_received = False
        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic, sensor_msgs.msg.CameraInfo, self.camera_info_callback)

    def __contains__(self, id):
        return id in self.simulator_entity_id_map or id in self.simulator_joint_id_map

    def load_urdf(self, id, filename, t, q):
        self.simulator_entity_id_map[id] = p.loadURDF(filename, t, q, flags=p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # If robot successfully loaded
        base_link_sim_id = self.simulator_entity_id_map[id]
        if base_link_sim_id > 0:
            # Create a joint map to ease exploration
            self.reverse_simulator_entity_id_map[base_link_sim_id] = id
            self.simulator_joint_id_map[base_link_sim_id] = {}
            self.reverse_simulator_joint_id_map[base_link_sim_id] = {}
            for i in range(0, p.getNumJoints(base_link_sim_id)):
                info = p.getJointInfo(base_link_sim_id, i)
                joint_name = info[1]
                self.simulator_joint_id_map[base_link_sim_id][info[1]] = i
                self.reverse_simulator_joint_id_map[base_link_sim_id][i] = info[1]
        else:
            raise ValueError("Invalid URDF file '{}' provided".format(robot_urdf_file_path))

    def update_entity(self, id, t, q):
        base_link_sim_id = self.simulator_entity_id_map[id]
        aabb_min, aabb_max = p.getAABB(base_link_sim_id)
        # if len(p.getOverlappingObjects(aabb_min, aabb_max)) == 0 and len(p.getContactPoints(base_link_sim_id)) == 0:
        #     # no overlapping and no contact, assume in the air ?
        #     p.resetBasePositionAndOrientation(base_link_sim_id, t, q)
        #     self.entities_state[id] = ObjectPhysicalState.INCONSISTENT
        # else:
        # t_current, q_current = p.getBasePositionAndOrientation(base_link_sim_id)
        # update_position = not np.allclose(np.array(t_current), t, atol=self.position_tolerance)
        # update_orientation = not np.allclose(np.array(q_current), q, atol=self.position_tolerance)
        # if update_position is True or update_orientation is True:
        p.resetBasePositionAndOrientation(base_link_sim_id, t, q)
        self.entities_state[id] = ObjectPhysicalState.PLAUSIBLE
        self.entities_last_update[id] = rospy.Time()

    def get_entity_position(self, id):
        entity_sim_id = self.simulator_entity_id_map[id]
        t, q = p.getBasePositionAndOrientation(entity_sim_id)
        return t, q

    def get_entity_velocity(self, id):
        entity_sim_id = self.simulator_entity_id_map[id]
        linear, angular = p.getBaseVelocity(entity_sim_id)
        return linear, angular

    def joint_states_callback(self, joint_states_msg):
        success, t, q = self.get_last_transform_from_tf2(self.global_frame_id, self.base_frame_id)
        if success is True:
            if self.robot_loaded is False:
                self.load_urdf(self.base_frame_id, self.robot_urdf_file_path, t, q)
                self.robot_loaded = True
            self.update_entity(self.base_frame_id, t, q)
        if self.robot_loaded is True:
            base_link_sim_id = self.simulator_entity_id_map[self.base_frame_id]
            for joint_index in range(0, len(joint_states_msg.name)):
                joint_sim_id = self.simulator_joint_id_map[base_link_sim_id][joint_states_msg.name[joint_index]]
                current_position, current_velocity, _, _ = p.getJointState(base_link_sim_id, joint_sim_id)
                update_position = not np.allclose(current_position, joint_states_msg.position[joint_index], atol=self.position_tolerance)
                update_velocity = not np.allclose(current_velocity, joint_states_msg.velocity[joint_index], atol=self.velocity_tolerance)
                if update_position is True or update_velocity is True:
                    p.resetJointState(base_link_sim_id, joint_sim_id, joint_states_msg.position[joint_index], joint_states_msg.velocity[joint_index])

    def camera_info_callback(self, msg):
        if self.camera_info_received is False:
            rospy.loginfo("Camera info received !")
            self.camera_info_received = True
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id

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

    def save(self):
        if self.last_backup_id is not None:
            p.removeState(self.last_backup_id)
        self.last_backup_id = p.saveState()

    def rollback(self):
        p.restoreState(self.last_backup_id)

    def get_robot_visibilities(self):
        if self.camera_info_received is True:
            pass
        else:
            return False, []

    def compose(pose1, pose2):
        t1, q1 = pose1
        t2, q2 = pose2
        return p.multiplyTransforms(t1, q1, t2, q2)

    def inverse_compose(pose1, pose2):
        t1, q1 = pose1
        t1, q1 = p.invertTransform(t1, q1)
        t2, q2 = pose2
        return p.multiplyTransforms(t1, q1, t2, q2)

    def get_visible_entities_bbox(self):
        pass

    def step_simulation(self):
        p.stepSimulation()

    def attach(self, parent_id, child_id):
        pass

    def dettach(self, parent_id, child_id):
        pass

    def get_robot_reachability(self):
        pass

    # def get_pointcloud_from_camera(self, t, q, max_range, width, height):
    #     rotation_matrix = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
    #     #print(rotation_matrix)
    #     rotated_frame = rotation_matrix.dot(np.eye(3))
    #     forward_vector = rotated_frame[:,0]
    #     yaw_vector = rotated_frame[:,2]
    #     camera_target = np.array(t) + forward_vector * max_range
    #
    #     #view_matrix = p.computeViewMatrix(t, camera_target, yaw_vector)
    #     #inv_view_matrix = np.linalg.inv(np.reshape(view_matrix, (4,4)))
    #
    #     fov = 60.0
    #     aspect = 1.3333
    #     clipnear = 0.3
    #     clipfar = 100.0
    #     projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, clipnear, clipfar)
    #     camera_image = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)
    #     # TODO: change??
    #     depth_tiny = camera_image[3] #(self.clipfar * self.clipnear) / (self.clipfar - (self.clipfar - self.clipnear) *
    #
    #     point_clouds = {}
    #
    #     pinhole_camera_model = PinholeCameraModel()
    #     pinhole_camera_model.fromCameraInfo(self.camera_info)
    #
    #     for v in range(height):
    #         #v_norm = v / float(height)
    #         for u in range(width):
    #             #u_norm = u / float(width)
    #             object_id = camera_image[4][v, u]
    #             pt3d = np.array(pinhole_camera_model.projectPixelTo3dRay((u, v))) * depth_tiny[v, u]
    #
    #             if u == width / 2. and v == height / 2.:
    #                 center_element = object_id
    #                 center_point = pt3d
    #                 # Pybullet call
    #                 #p.addUserDebugLine(t, [pt3d[0], pt3d[1], pt3d[2]], [1, 0, 0])
    #                 continue
    #             else:
    #                 # Pybullet call
    #                 #p.addUserDebugLine(t, [pt3d[0], pt3d[1], pt3d[2]], [0, 0, 1])
    #
    #             if object_id not in point_clouds.keys():
    #                 point_clouds[object_id] = []
    #
    #             point_clouds[object_id].append(list(pt3d) + [1]) # 1 is intensity
    #
    #     return point_clouds, center_element, center_point
    #
    def get_human_visibilities(self, t, q, focus_distance=1.0):
        if self.camera_info is not None:
            visibilities = np.array([])
            focus_3d_pt = None
            #flag, visibilities, _ = self.get_pointcloud_from_camera(t, q, 10, 8, 8)
            euler_angles = euler_from_quaternion(q, "rxyz")
            #view_matrix = inverse_matrix(compose_matrix(angles=euler_angles, translate=t))
            #view_matrix = p.computeViewMatrixFromYawPitchRoll(t, 1.0, euler_angles[2], euler_angles[1], euler_angles[0], 2)
            view_matrix = p.computeViewMatrixFromYawPitchRoll([0,0,1], 1.0, 0, 0, 0, 2)
            projection_matrix = p.computeProjectionMatrixFOV(HumanVisualModel.FOV,
                                                             HumanVisualModel.ASPECT,
                                                             HumanVisualModel.CLIPNEAR,
                                                             HumanVisualModel.CLIPFAR)
            # pinhole_camera_model = PinholeCameraModel()
            # pinhole_camera_model.fromCameraInfo(self.camera_info)
            # print view_matrix
            #camera_image = p.getCameraImage(HumanVisualModel.WIDTH, HumanVisualModel.HEIGHT, viewMatrix=view_matrix, projectionMatrix=projection_matrix)#, renderer=p.ER_TINY_RENDERER)
            camera_image = p.getCameraImage(HumanVisualModel.WIDTH, HumanVisualModel.HEIGHT, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
            # depth_map = camera_image[3]
            # points = []
            # center_u = int(HumanVisualModel.WIDTH/2.)
            # center_v = int(HumanVisualModel.HEIGHT/2.)
            # center_point = np.array(pinhole_camera_model.projectPixelTo3dRay((center_u, center_v))) * depth_map[center_v, center_u]
            # viz_frame = np.zeros((HumanVisualModel.HEIGHT,HumanVisualModel.WIDTH,3), np.uint8)
            # for v in range(HumanVisualModel.HEIGHT):
            #     #v_norm = v / float(height)
            #     for u in range(HumanVisualModel.WIDTH):
            #         if u == center_u and v == center_v:
            #             continue
            #         object_id = camera_image[4][v, u]
            #         pt3d = np.array(pinhole_camera_model.projectPixelTo3dRay((u, v))) * depth_map[v, u]
            #         points.append(pt3d)
            #         d = math.sqrt(np.sum((center_point - pt3d)**2)) # Euler distance
            #         d_clipped = min(d, focus_distance)
            #         d_norm = d_clipped / focus_distance
            #         intensity = 1.0 - d_norm
            #         (r, g, b) = colorsys.hsv_to_rgb(min(1.0-intensity, 0.8333), 1.0, 1.0)
            #         viz_frame[v, u] = (b*255, g*255, r*255)
            # print(viz_frame.shape)
            # print(center_point)
            # print(len(points))

            viz_img_msg = self.bridge.cv2_to_imgmsg(camera_image[3])
            self.visualization_publisher.publish(viz_img_msg)
            return False, []
        else:
            return False, []
