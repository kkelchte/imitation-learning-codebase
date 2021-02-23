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


class WaypointIndicator:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/robot/position_sensor') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        # fields
        self._current_waypoint_index = 0
        self._waypoints = rospy.get_param('/world/waypoints', [])
        self._waypoint_reached_distance = rospy.get_param('/world/waypoint_reached_distance', 0.3)
        self._local_waypoints = rospy.get_param('/world/express_waypoints_locally', False)
        self.robot_pose = None

        # publishers
        self._publisher = rospy.Publisher('/waypoint_indicator/current_waypoint', Float32MultiArray, queue_size=10)

        # subscribe
        odometry_type = rospy.get_param('/robot/position_sensor/type')
        callback = f'_{camelcase_to_snake_format(odometry_type)}_callback'
        assert callback in self.__dir__()
        rospy.Subscriber(name=rospy.get_param('/robot/position_sensor/topic'),
                         data_class=eval(odometry_type),
                         callback=eval(f'self.{callback}'))
        rospy.Subscriber(name='/fsm/reset',
                         data_class=Empty,
                         callback=self.reset)
        self._fsm_state = FsmState.Unknown
        rospy.Subscriber(name='/fsm/state', data_class=String, callback=self._set_fsm_state)

        rospy.init_node('waypoint_indicator')
        self.reset()

    def _set_fsm_state(self, msg: String):
        # detect transition
        if self._fsm_state != FsmState[msg.data]:
            self._fsm_state = FsmState[msg.data]
            if self._fsm_state == FsmState.Running and self._local_waypoints:  # reset early feedback values
                relative_points = [np.asarray([wp[0], wp[1], wp[2]]) if len(wp) == 3
                                   else np.asarray([wp[0], wp[1], self.robot_pose.position.z])
                                   for wp in self._waypoints]
                absolute_points = transform(relative_points, self.robot_pose.orientation, self.robot_pose.position)
                self._waypoints = absolute_points

    def reset(self, msg: Empty = None):
        self._waypoints = rospy.get_param('/world/waypoints', [])
        self._waypoints = [[float(coor) for coor in wp] for wp in self._waypoints]
        self._current_waypoint_index = 0
        self._waypoint_reached_distance = rospy.get_param('/world/waypoint_reached_distance', 0.5)
        if len(self._waypoints) == 0:
            cprint(message='could not find waypoints in rosparam so exit',
                   logger=self._logger,
                   msg_type=MessageType.error)
            sys.exit(0)
        else:
            cprint(f'waypoints: {self._waypoints}', self._logger)

    def _odometry_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose
        # adjust orientation towards current_waypoint
        dy = (self._waypoints[self._current_waypoint_index][1] - self.robot_pose.position.y)
        dx = (self._waypoints[self._current_waypoint_index][0] - self.robot_pose.position.x)

        if np.sqrt(dx ** 2 + dy ** 2) < self._waypoint_reached_distance:
            # update to next waypoint:
            self._current_waypoint_index += 1
            self._current_waypoint_index = self._current_waypoint_index % len(self._waypoints)
            cprint(f"Reached waypoint: {self._waypoints[self._current_waypoint_index-1]}, "
                   f"next waypoint @ {self._waypoints[self._current_waypoint_index]}.", self._logger)
            self._adjust_yaw_waypoint_following = 0

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self._fsm_state == FsmState.Running:
                current_waypoint = self._waypoints[self._current_waypoint_index]
                multi_array = Float32MultiArray()
                multi_array.data = current_waypoint
                self._publisher.publish(multi_array)
            rate.sleep()


if __name__ == "__main__":
    waypoint_indicator = WaypointIndicator()
    waypoint_indicator.run()
