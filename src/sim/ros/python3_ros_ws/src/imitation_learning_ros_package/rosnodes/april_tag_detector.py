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
from geometry_msgs.msg import Transform, PointStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty, String
from tf2_msgs.msg import TFMessage
from tf2_ros.transform_listener import TransformListener

from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import camelcase_to_snake_format, get_filename_without_extension
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, transform


class AprilTagDetector:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/robot/position_sensor') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        # fields
        self._detected_waypoints = {}
        self._waypoint_reached_distance = rospy.get_param('/world/waypoint_reached_distance', 0.3)


        # publishers
        self._publisher = rospy.Publisher('/reference_ground_point', PointStamped, queue_size=10)

        # subscribe
        rospy.Subscriber(name='/tf',
                         data_class=TFMessage,
                         callback=self._set_tag_transforms)
        rospy.init_node('april_tag_detector')

    def _set_tag_transforms(self, msg: TFMessage):
        for transform in msg.transforms:
            if 'tag' in transform.child_frame_id:
                distance = sum([transform.transform.translation.x**2,
                                transform.transform.translation.y**2])  # , transform.transform.translation.z**2
                self._detected_waypoints[distance] = transform.transform.translation

    def _select_next_waypoint_transform(self) -> Transform.translation:
        distances = list(self._detected_waypoints.keys())
        distances.sort()
        while distances[0] < self._waypoint_reached_distance:
            distances.pop(0)
        return self._detected_waypoints[distances[0]]

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self._detected_waypoints != {}:
                closest_tag_transform = self._select_next_waypoint_transform()
                reference_point = PointStamped(point=Point(
                    x=closest_tag_transform.x,
                    y=closest_tag_transform.y,
                    z=closest_tag_transform.z
                )
                )
                reference_point.header.frame_id = 'camera_optical'
                # use tags only as reference points
                self._detected_waypoints = {}
                self._publisher.publish(reference_point)
            rate.sleep()


if __name__ == "__main__":
    waypoint_indicator = AprilTagDetector()
    waypoint_indicator.run()
