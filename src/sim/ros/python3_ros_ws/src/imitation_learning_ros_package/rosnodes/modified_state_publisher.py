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

        self._mode = rospy.get_param('/modified_state_publisher/mode')
        cprint(f'mode: {self._mode}', self._logger)
        rospy.Subscriber(rospy.get_param('/fsm/state_topic', '/fsm/state'), String, self._fsm_state_update)

        if self._mode == 'global_poses':
            self._publisher = rospy.Publisher(rospy.get_param('/robot/modified_state_topic', '/modified_state'),
                                              CombinedGlobalPoses, queue_size=10)
            self.tracking_position = np.asarray([999.]*3)
            self.tracking_orientation = np.asarray([999.]*3)
            self.fleeing_position = np.asarray([999.]*3)
            for agent in ['tracking',
                          'fleeing']:
                topic_name = rospy.get_param(f'/robot/tf_{agent}_topic', f'/{agent}/ground_truth_to_tf/pose')
                type_name = rospy.get_param(f'/robot/tf_{agent}_type', 'PoseStamped')
                rospy.Subscriber(topic_name, eval(type_name), self._set_pose, callback_args=agent)
                cprint(f'subscribing to: {topic_name}', self._logger, msg_type=MessageType.debug)
        else:
            raise NotImplemented(f'[modified_state_publisher]: type of mode for modification is not implemented: '
                                 f'{self._mode}')
        self._fsm_state = FsmState.Unknown
        self._rate_fps = rospy.get_param('/modified_state_publisher/rate_fps', 60)
        rospy.init_node('modified_state_publisher')

    def _fsm_state_update(self, msg: String):
        if self._fsm_state != FsmState[msg.data]:
            cprint(f'update fsm state to {FsmState[msg.data]}', self._logger, msg_type=MessageType.debug)
        self._fsm_state = FsmState[msg.data]

    def _set_pose(self, msg, args):
        agent = args
        if isinstance(msg, PoseStamped):
            pose = process_pose_stamped(msg)
            cprint(f'received pose of agent {agent} with value {pose}', self._logger)
            if agent == 'tracking':
                self.tracking_position = pose[0:3]
                self.tracking_orientation = euler_from_quaternion(pose[3:])
            elif agent == 'fleeing':
                self.fleeing_position = pose[0:3]
            else:
                raise NotImplemented(f'[modified_state_publisher]: unknown agent {agent}')
            state = np.asarray([*self.tracking_position, *self.fleeing_position, *self.tracking_orientation])
            cprint(f'current state: {state}', self._logger)

    def publish(self):
        state = np.asarray([*self.tracking_position, *self.fleeing_position, *self.tracking_orientation])
        if 999 not in state:  # and self._fsm_state == FsmState.Running
            self._publisher.publish(array_to_combined_global_pose(state))

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


if __name__ == "__main__":
    publisher = ModifiedStatePublisher()
    publisher.run()
