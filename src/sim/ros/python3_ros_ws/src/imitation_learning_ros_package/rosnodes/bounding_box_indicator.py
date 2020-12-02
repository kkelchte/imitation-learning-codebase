#!/usr/bin/python3.8

"""Extract waypoints from rosparam provided by config/world/...
Keep track of current position and next waypoint.

extension 1:
keep track of 2d (or 3d) poses to map trajectory in a topdown view.
"""
import os
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty

from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension
from src.sim.ros.src.utils import get_output_path, process_pose_stamped, process_odometry


class BoundingBoxIndicator:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/robot/odometry_topic') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        # fields
        self.agent_poses = {'tracking': None, 'fleeing': None}

        # publishers
        self._publisher = rospy.Publisher('/waypoint_indicator/current_waypoint', Float32MultiArray, queue_size=10)

        # subscribe
        self._subscribe()
        rospy.init_node('waypoint_indicator')
        self.reset()

    def _subscribe(self):
        rospy.Subscriber(name=rospy.get_param('/robot/tracking_tf_topic'),
                         data_class=eval(rospy.get_param('/robot/tracking_tf_type')),
                         callback=self._set_field,
                         callback_args='tracking')
        rospy.Subscriber(name=rospy.get_param('/robot/fleeing_tf_topic'),
                         data_class=eval(rospy.get_param('/robot/fleeing_tf_type')),
                         callback=self._set_field,
                         callback_args='fleeing')
        rospy.Subscriber(name=rospy.get_param('/fsm/reset_topic', '/reset'),
                         data_class=Empty,
                         callback=self.reset)

    def _set_field(self, msg, args):
        agent_name = args
        if isinstance(msg, PoseStamped):
            self.agent_poses[agent_name] = process_pose_stamped(msg)
        elif isinstance(msg, Odometry):
            self.agent_poses[agent_name] = process_odometry(msg)
        else:
            raise NotImplemented(f'failed to process message of type {type(msg)}')

    def reset(self, msg: Empty = None):
        self.agent_poses = {'tracking': None, 'fleeing': None}

    def _calculate_and_publish(self):
        return None

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self._calculate_and_publish()
            rate.sleep()


if __name__ == "__main__":
    bounding_box_indicator = BoundingBoxIndicator()
    bounding_box_indicator.run()
