#!/usr/bin/python3.8

"""Extract waypoints from rosparam provided by config/world/...
Keep track of current position and next waypoint.

extension 1:
keep track of 2d (or 3d) poses to map trajectory in a topdown view.
"""
import copy
import os
import sys
import time
from typing import Any, Union

import numpy as np
import rospy
from geometry_msgs.msg import Transform, PointStamped, Point, TransformStamped, PoseStamped, PoseWithCovarianceStamped, \
    Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Empty, String
from tf2_msgs.msg import TFMessage
from apriltag_ros.msg import AprilTagDetectionArray

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
        self._optical_to_base_transformation_dict = {
            '/forward/image_raw': {'translation': [0, -0.086, -0.071],
                                   'rotation': [0.553, -0.553, 0.440, 0.440]},
            '/down/image_raw': {'translation': [-0.000, 0.050, -0.100],
                                'rotation': [0.707, -0.707, 0.000, 0.000]},
            '/bebop/image_raw': {'translation': [0., 0.022, -0.097],
                                 'rotation': [0.553, -0.553, 0.44, 0.44]},
        }
        self._optical_to_base_transformation = self._optical_to_base_transformation_dict[
            rospy.get_param('/robot/camera_sensor/topic')
        ]
        self._tag_transforms = {}
        self._waypoint_reached_distance = rospy.get_param('/world/waypoint_reached_distance', 0.3)
        self._calculating = False  # Don't update tag detections while calculating closest
        # publishers
        self._publisher = rospy.Publisher('/reference_pose', PointStamped, queue_size=10)

        # subscribe
        rospy.Subscriber(name='/tag_detections',
                         data_class=AprilTagDetectionArray,
                         callback=self._set_tag_transforms)
        rospy.Subscriber(name='/bebop/camera_control', data_class=Twist, callback=self._adjust_for_camera_twist)
        # start node
        rospy.init_node('april_tag_detector')
        cprint(f"Initialised", logger=self._logger)

    def _adjust_for_camera_twist(self, msg: Twist):
        if msg.angular.y == -90:
            self._optical_to_base_transformation['rotation'] = [0.707, -0.707, 0.000, 0.000]
        elif msg.angular.y == 0:
            self._optical_to_base_transformation['rotation'] = [0., 0., 0., 1.0]
        else:
            self._optical_to_base_transformation['rotation'] = [0.553, -0.553, 0.44, 0.44]

    def _set_tag_transforms(self, msg: AprilTagDetectionArray):
        if self._calculating:
            return
        for detection in msg.detections:
            self._tag_transforms[detection.id] = detection.pose

    def _select_next_waypoint_transform(self) -> Union[PointStamped, None]:
        # transform detected tags to agents base_link
        tags_in_base_link = {}
        for k in self._tag_transforms.keys():
            tags_in_base_link[k] = transform(
                points=[self._tag_transforms[k].pose.pose.position],
                orientation=self._optical_to_base_transformation['rotation'],
                translation=np.asarray(self._optical_to_base_transformation['translation']),
                invert=True
            )[0]
        print(f'tags_in_base_link: {tags_in_base_link}')
        # for all tag transforms lying in front (x>0 in base_link frame), measure the distance
        distances = {
            k: np.sqrt(sum([tags_in_base_link[k].x**2,
                            tags_in_base_link[k].y**2]))
            for k in self._tag_transforms.keys() if tags_in_base_link[k].x > 0
        }
        # ignore tags that are closer than the 'distance-reached'
        further_distances = {k: distances[k] for k in distances.keys()
                             if distances[k] > self._waypoint_reached_distance}
        print(f'distances: {distances}')
        # TODO: ignore tags with a too large covariance

        # sort tags and take closest
        sorted_distances = dict(sorted(further_distances.items(), key=lambda item: item[1]))
        if len(sorted_distances) == 0:
            return None
        else:
            tag_id = list(sorted_distances.keys())[0]
        # create a PointStamped message
        print(f'tag_id: {tag_id}')
        reference_point = PointStamped(point=Point(
            x=tags_in_base_link[tag_id].x,
            y=tags_in_base_link[tag_id].y,
            z=0  # as tag lies probably on the ground just fly towards and above it
        ))
        reference_point.header.frame_id = 'agent'
        reference_point.header.stamp = self._tag_transforms[tag_id].header.stamp
        return reference_point

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self._calculating = True
            next_reference_point = self._select_next_waypoint_transform()
            self._calculating = False
            if next_reference_point is not None:
                cprint(f"reference tag: {next_reference_point}", logger=self._logger)
                self._publisher.publish(next_reference_point)
                self._tag_transforms = {}
            else:
                cprint(f"detected tags: {self._tag_transforms.keys()}", logger=self._logger)
            rate.sleep()


if __name__ == "__main__":
    waypoint_indicator = AprilTagDetector()
    waypoint_indicator.run()
