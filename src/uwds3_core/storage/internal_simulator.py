import rospy
import numpy as np
import pybullet as p
import pybullet_data
import time
import tf2_ros
import geometry_msgs
import sensor_msgs
from tf2_ros import Buffer, TransformListener
from pybullet import JOINT_FIXED


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
            self.simulator_entity_id_map["env"] = p.loadSDF(self.env_sdf_file_path)

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
        t_current, q_current = p.getBasePositionAndOrientation(base_link_sim_id)
        update_position = not np.allclose(t_current, t, atol=self.position_tolerance)
        update_orientation = not np.allclose(q_current, q, atol=self.position_tolerance)
        if update_position is True or update_orientation is True:
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

    def get_human_visibilities(self, id, focus_distance):
        camera_position = self.get_entity_position(id)
        if self.camera_info_received is True:
            pass
        else:
            return False, []
