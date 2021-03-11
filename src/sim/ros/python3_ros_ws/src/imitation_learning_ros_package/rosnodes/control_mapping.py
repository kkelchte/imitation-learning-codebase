#!/usr/bin/python3.8
"""Listens to FSM state and changes controls accordingly

Configuration defines which FSM state corresponds to which control connection
Each FSM state has a number of potential actors or controls steering the robot:
Running -> EXPERT / DNN / USER < CONFIG/ACTOR & CONFIG/SUPERVISOR
TakeOver -> USER
DriveBack -> DB

Config defines for required FSM states the corresponding topics to be connected to the robot cmd_vel.

Extension 1:
Add stepbased implementation by following the frame rate, assuming a certain control rate,
if period between controls is surpassed bring drone to hover mode.
max_time = 1 / 5.  # max time should be around the 1/(rgb frame rate) (1/10FPS)
(set dynamically depending on the control rate)
current_time = 0.  # last time step got from gazebo or time
control_time = 0.  # time in seconds of moment of last send control

Extension 2:
Adjust height automatically or allow one actor to control only several twist arguments
Control height
aggressiveness = 1.  # define how direct the yaw turn is put on control
adjust_height = -100
"""
import os
import time
from copy import deepcopy

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension
from src.sim.common.noise import *  # Do not remove
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, apply_noise_to_twist


class ControlMapper:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/control_mapping/mapping') and time.time() < stime + max_duration:
            time.sleep(0.1)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        self._mapping = rospy.get_param('/control_mapping/mapping')
        cprint(f'mapping: {self._mapping}', self._logger)

        # get robot command topics according to control keys
        self._publishers = {}  # required publishers to control robot(s)
        control_types = []
        for fsm_state in FsmState.members():
            if fsm_state not in self._mapping.keys():
                self._mapping[fsm_state] = {}
            else:
                control_types.extend(self._mapping[fsm_state].keys())
        for control_type in set(control_types):
            self._publishers[control_type] = rospy.Publisher(rospy.get_param(f'/robot/{control_type}_topic'),
                                                             Twist, queue_size=10)

        noise_config = rospy.get_param('/control_mapping/noise', None)
        self._noise = eval(f"{noise_config['name']}(**noise_config['args'])") if noise_config is not None else None
        self._fsm_state = FsmState.Unknown
        self._messages = {}  # contains all controls comin from actors
        self._rate_fps = rospy.get_param('/control_mapping/rate_fps', 60)
        self.count = 0
        self._subscribe()
        rospy.init_node('control_mapper')

    def _subscribe(self):
        rospy.Subscriber('/fsm/state', String, self._fsm_state_update)
        # For each actor add subscriber < actor config
        actor_topics = []
        for state, controls in self._mapping.items():
            for ctr in controls.values():
                actor_topics.append(ctr)
        actor_topics = set(actor_topics)
        cprint(f'subscribing to: {actor_topics}', self._logger, msg_type=MessageType.debug)
        for topic in actor_topics:
            rospy.Subscriber(topic, Twist, self._control_callback, callback_args=topic)

    def _fsm_state_update(self, msg: String):
        self.count = 0
        if self._fsm_state != FsmState[msg.data]:
            cprint(f'update fsm state to {FsmState[msg.data]}', self._logger, msg_type=MessageType.debug)
        self._fsm_state = FsmState[msg.data]
        if self._fsm_state.name not in self._mapping.keys():
            raise KeyError(f'Unrecognised Fsm state {self._fsm_state}.\n Not in current mapping: {self._mapping}.\n'
                           f'Ensure all FSM States are specified in config/control_mapper/*.yml.')

    def _control_callback(self, msg: Twist, topic_name):
        self._messages[topic_name] = msg

    def publish(self):
        for control_type, actor_topic in self._mapping[self._fsm_state.name].items():
            if actor_topic in self._messages.keys():
                control_msg = self._messages[actor_topic]
                if self._noise is not None:
                    control_msg = apply_noise_to_twist(twist=deepcopy(control_msg),
                                                       noise=self._noise.sample())
                self._publishers[control_type].publish(control_msg)

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self.publish()
            # TODO extension 1 (see up)
            # if (current_time - control_time) > max_time:
            #     self.cmd_pub.publish(Twist())
            #     control_time = current_time
            rate.sleep()
            self.count += 1
            if self.count % 1 * self._rate_fps == 0:
                msg = f"{rospy.get_time(): 0.0f}ms:"
                msg += f" state: {FsmState(self._fsm_state).name} "
                for actor_topic in self._messages.keys():
                    msg += f" {actor_topic} {self._messages[actor_topic]}\n"
                msg += f" publishing on "
                for robot_control in self._mapping[self._fsm_state.name].keys():
                    msg += f" {robot_control},"
                cprint(msg, self._logger, msg_type=MessageType.debug)


if __name__ == "__main__":
    control_mapper = ControlMapper()
    control_mapper.run()
