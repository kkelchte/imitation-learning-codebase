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
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty, String

from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, transform


class ChessboardDetector:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/robot/position_sensor') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        # fields
        self.robot_pose = None
        self._detected_waypoint = None

        # publishers
        self._publisher = rospy.Publisher('/waypoint_indicator/current_waypoint', Float32MultiArray, queue_size=10)

        # subscribe
        odometry_type = rospy.get_param('/robot/position_sensor/type')
        callback = f'_{camelcase_to_snake_format(odometry_type)}_callback'
        assert callback in self.__dir__()
        rospy.Subscriber(name=rospy.get_param('/robot/position_sensor/topic'),
                         data_class=eval(odometry_type),
                         callback=eval(f'self.{callback}'))
        self._fsm_state = FsmState.Unknown
        rospy.Subscriber(name='/fsm/state', data_class=String, callback=self._set_fsm_state)

        rospy.init_node('chessboard_detector')

    def _set_fsm_state(self, msg: String):
        # detect transition
        if self._fsm_state != FsmState[msg.data]:
            self._fsm_state = FsmState[msg.data]

    def _odometry_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self._fsm_state == FsmState.Running and self._detected_waypoint is not None:
                multi_array = Float32MultiArray()
                self._publisher.publish(multi_array)
            rate.sleep()


if __name__ == "__main__":
    waypoint_indicator = ChessboardDetector()
    waypoint_indicator.run()
