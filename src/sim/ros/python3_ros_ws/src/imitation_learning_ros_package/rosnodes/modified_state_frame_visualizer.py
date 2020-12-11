#!/usr/bin/python3.8
"""
Modifies sensor information to a state or observation used by rosenvironment.
Defines separate modes:
Default mode:
    combine tf topics of tracking and fleeing agent in global frame
"""
import os
import time
from copy import deepcopy

import numpy as np
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Float32MultiArray

from src.core.data_types import SensorType
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension, camelcase_to_snake_format, ros_message_to_type_str
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from imitation_learning_ros_package.msg import CombinedGlobalPoses
from src.sim.ros.src.utils import process_odometry, process_pose_stamped, euler_from_quaternion, get_output_path, \
    array_to_combined_global_pose, calculate_bounding_box


class ModifiedStateFrameVisualizer:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/modified_state_publisher/mode') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        self._mode = rospy.get_param('/modified_state_publisher/mode', 'CombinedGlobalPoses')
        cprint(f'mode: {self._mode}', self._logger)
        rospy.Subscriber(rospy.get_param('/robot/modified_state_sensor/topic', '/modified_state'),
                         eval(rospy.get_param('/robot/modified_state_sensor/type', 'CombinedGlobalPoses')),
                         self._process_state_and_publish_frame)
        self._publisher = rospy.Publisher('/modified_state_frame',
                                          Float32MultiArray, queue_size=10)
        rospy.init_node('modified_state_frame_visualizer')

    def _publish_combined_global_poses(self, data: np.ndarray) -> None:
        resolution = (100, 100)
        position, width, height = calculate_bounding_box(state=data,
                                                         resolution=resolution)
        frame = np.zeros(resolution)
        frame[position[0]:position[0] + width,
        position[1]:position[1] + height] = 1
        self._publisher.publish(frame)

    def _process_state_and_publish_frame(self, msg: CombinedGlobalPoses):
        msg_type = camelcase_to_snake_format(ros_message_to_type_str(msg))
        data = eval(f'process_{msg_type}(msg)')
        if msg_type == 'combined_global_poses':
            self._publish_combined_global_poses(data)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    publisher = ModifiedStateFrameVisualizer()
    publisher.run()
