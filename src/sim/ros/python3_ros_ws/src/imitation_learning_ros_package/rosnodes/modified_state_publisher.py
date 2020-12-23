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
from std_msgs.msg import String

from src.core.data_types import SensorType
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension, camelcase_to_snake_format
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from imitation_learning_ros_package.msg import CombinedGlobalPoses
from src.sim.ros.src.utils import process_odometry, process_pose_stamped, euler_from_quaternion, get_output_path, \
    array_to_combined_global_pose


class ModifiedStatePublisher:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/modified_state_publisher/mode') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        self._mode = rospy.get_param('/modified_state_publisher/mode', 'CombinedGlobalPoses')
        cprint(f'mode: {self._mode}', self._logger)
        self._fsm_state = FsmState.Unknown
        self._rate_fps = rospy.get_param('/modified_state_publisher/rate_fps', 60)
        rospy.Subscriber('/fsm/state', String, self._fsm_state_update)
        self._setup()
        rospy.init_node('modified_state_publisher')

    def _setup(self):
        if self._mode == 'CombinedGlobalPoses':
            self._publisher = rospy.Publisher(rospy.get_param('/robot/modified_state_topic', '/modified_state'),
                                              CombinedGlobalPoses, queue_size=10)
            self.tracking_position = np.asarray([999.]*3)
            self.tracking_orientation = np.asarray([999.]*3)
            self.fleeing_position = np.asarray([999.]*3)
            for sensor_type in [SensorType.tracking_position,
                                SensorType.fleeing_position]:
                if rospy.has_param(f'/robot/{sensor_type.name}_sensor'):
                    msg_type = rospy.get_param(f'/robot/{sensor_type.name}_sensor/type')
                    rospy.Subscriber(name=rospy.get_param(f'/robot/{sensor_type.name}_sensor/topic'),
                                     data_class=eval(msg_type),
                                     callback=eval(f"self._set_{camelcase_to_snake_format(msg_type)}"),
                                     callback_args=sensor_type)
                    cprint(f'subscribing to: {rospy.get_param(f"/robot/{sensor_type.name}_sensor/topic")}',
                           self._logger, msg_type=MessageType.debug)
        else:
            raise NotImplemented(f'[modified_state_publisher]: type of mode for modification is not implemented: '
                                 f'{self._mode}')

    def _fsm_state_update(self, msg: String):
        if self._fsm_state != FsmState[msg.data]:
            cprint(f'update fsm state to {FsmState[msg.data]}', self._logger, msg_type=MessageType.debug)
        self._fsm_state = FsmState[msg.data]

    def _set_pose_stamped(self, msg, args):
        sensor_type = args
        pose = process_pose_stamped(msg)
        if sensor_type == SensorType.tracking_position:
            self.tracking_position = pose[0:3]
            self.tracking_orientation = euler_from_quaternion(pose[3:])
        elif sensor_type == SensorType.fleeing_position:
            self.fleeing_position = pose[0:3]
        else:
            raise NotImplemented(f'[modified_state_publisher]: unknown type {sensor_type}')

    def publish(self):
        state = np.asarray([*self.tracking_position, *self.fleeing_position, *self.tracking_orientation])
        if 999 not in state:
            self._publisher.publish(array_to_combined_global_pose(state))

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


if __name__ == "__main__":
    publisher = ModifiedStatePublisher()
    publisher.run()
