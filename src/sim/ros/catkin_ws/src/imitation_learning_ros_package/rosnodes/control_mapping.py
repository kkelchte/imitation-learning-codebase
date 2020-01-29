#!/usr/bin/python3.7
"""Listens to FSM state and changes controls accordingly

Configuration defines which FSM state corresponds to which control connection
Each FSM state has a number of potential actors or controls steering the robot:
Running -> EXPERT / DNN / USER < CONFIG/ACTOR & CONFIG/SUPERVISOR
TakeOff -> EXPERT
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

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from src.core.logger import get_logger, cprint
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path


class ControlMapper:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/control_mapping/mapping') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(os.path.basename(__file__), self._output_path)

        self._mapping = rospy.get_param('/control_mapping/mapping')
        cprint(f'mapping: {self._mapping}', self._logger)
        for key in FsmState.members():
            if key not in self._mapping.keys():
                self._mapping[key] = {}
        self._fsm_state = FsmState.Unknown

        self._publishers = {
            'command': rospy.Publisher(rospy.get_param('/robot/command_topic'), Twist, queue_size=10),
            'supervision': rospy.Publisher(rospy.get_param('/control_mapping/supervision_topic'),
                                           Twist, queue_size=10)
        }

        self._subscribe()
        rospy.init_node('control_mapper')

    def _subscribe(self):
        rospy.Subscriber(rospy.get_param('/fsm/state_topic', '/fsm_state'), String, self._fsm_state_update)
        # For each actor add subscriber < actor config
        actor_topics = []
        for state, mode in self._mapping.items():
            if 'command' in mode.keys():
                # if param (mode[command]) does not exist, assume mode[command] is the topic
                actor_topics.append(rospy.get_param(mode['command'], mode['command']))
            if 'supervision' in mode.keys():
                # if param (mode[supervision]) does not exist, assume mode[supervision] is the topic
                actor_topics.append(rospy.get_param(mode['supervision'], mode['supervision']))
        actor_topics = set(actor_topics)
        for topic in actor_topics:
            rospy.Subscriber(topic, Twist, self._control_callback, callback_args=topic)

    def _fsm_state_update(self, msg: String):
        if self._fsm_state != FsmState[msg.data]:
            cprint(f'update fsm state to {FsmState[msg.data]}', self._logger)
        self._fsm_state = FsmState[msg.data]
        if self._fsm_state.name not in self._mapping.keys():
            raise KeyError(f'Unrecognised Fsm state {self._fsm_state}.\n Not in current mapping: {self._mapping}.\n'
                           f'Ensure all FSM States are specified in config/control_mapper/*.yml.')

    def _control_callback(self, msg: Twist, topic_name):
        if 'command' in self._mapping[self._fsm_state.name].keys() \
                and topic_name == self._mapping[self._fsm_state.name]['command']:
            self._publishers['command'].publish(msg)
        if 'supervision' in self._mapping[self._fsm_state.name].keys() \
                and topic_name == self._mapping[self._fsm_state.name]['supervision']:
            self._publishers['supervision'].publish(msg)


def run():
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        # TODO extension 1 (see up)
        # if (current_time - control_time) > max_time:
        #     self.cmd_pub.publish(Twist())
        #     control_time = current_time
        rate.sleep()


if __name__ == "__main__":
    control_mapper = ControlMapper()
    run()
