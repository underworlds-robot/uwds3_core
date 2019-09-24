#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import tf2_ros
from tf2_ros import Buffer, TransformListener
import numpy as np
from sensor_msgs.msg import CameraInfo, Image, JointState
from uwds3_msgs.msg import SceneChangesStamped
from uwds_core.underworlds_core import UnderworldsCore
import message_filters
import sensor_msgs


class Uwds3CoreNode(object):
    def __init__(self):
        rospy.init_node("uwds3_core")
        rospy.loginfo("Starting Underworlds core...")
        self.underworlds_core = UnderworldsCore()
        rospy.loginfo("Underworlds core ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    core = Uwds3CoreNode().run()
