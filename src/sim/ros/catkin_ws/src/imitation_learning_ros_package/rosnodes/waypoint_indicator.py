#!/usr/bin/python3.7

"""Extract waypoints from rosparam provided by config/world/...
Keep track of current position and next waypoint.

extension 1:
keep track of 2d (or 3d) poses to map trajectory in a topdown view.
"""
import sys
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import camelcase_to_snake_format


class WaypointIndicator:

    def __init__(self):
        while not rospy.has_param('/robot/pose_estimation_topic'):
            time.sleep(0.1)
        self._logger = get_logger('waypoint_indicator', rospy.get_param('output_path'))
        self._waypoints = rospy.get_param('/world/waypoints', [])
        self._current_waypoint_index = 0
        # TODO extension: add automatic closest waypoint detection and start flying to there
        #  ==> this allows robot to be spawned anywhere on the trajectory.
        self._waypoint_reached_distance = rospy.get_param('/world/waypoint_reached_distance', 0.5)
        if len(self._waypoints) == 0:
            cprint(message='could not find waypoints in rosparam so exit',
                   logger=self._logger,
                   msg_type=MessageType.error)
            sys.exit(0)
        self._publisher = rospy.Publisher('/waypoint_indicator/current_waypoint', Float32MultiArray, queue_size=10)
        pose_estimation_type = rospy.get_param('/robot/pose_estimation_type')
        callback = f'_process_{camelcase_to_snake_format(pose_estimation_type)}'
        assert callback in self.__dir__()
        rospy.Subscriber(name=rospy.get_param('/robot/pose_estimation_topic'),
                         data_class=eval(pose_estimation_type),
                         callback=eval(f'self.{callback}'))
        rospy.init_node('waypoint_indicator')

    def _process_odometry(self, msg: Odometry):
        # adjust orientation towards current_waypoint
        dy = (self._waypoints[self._current_waypoint_index][1] - msg.pose.pose.position.y)
        dx = (self._waypoints[self._current_waypoint_index][0] - msg.pose.pose.position.x)

        if np.sqrt(dx ** 2 + dy ** 2) < self._waypoint_reached_distance:
            # update to next waypoint:
            self._current_waypoint_index += 1
            self._current_waypoint_index = self._current_waypoint_index % len(self._waypoints)
            cprint(f"Reached waypoint: {self._current_waypoint_index-1}, "
                   f"next waypoint @ {self._waypoints[self._current_waypoint_index]}.", self._logger)
            self._adjust_yaw_waypoint_following = 0

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            current_waypoint = self._waypoints[self._current_waypoint_index]
            multi_array = Float32MultiArray()
            multi_array.data = current_waypoint
            self._publisher.publish(multi_array)
            rate.sleep()


if __name__ == "__main__":
    waypoint_indicator = WaypointIndicator()
    waypoint_indicator.run()
