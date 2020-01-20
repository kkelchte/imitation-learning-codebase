#!/usr/bin/python3.7
import asyncio
from typing import Union

import roslib
import time
import numpy as np

from src.core.utils import camelcase_to_snake_format
from src.sim.common.actors import Actor
from src.sim.common.data_types import Action, State, SensorType, TerminalType
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import adapt_action_to_twist, process_compressed_image, process_image, process_laser_scan

roslib.load_manifest('imitation_learning_ros_package')
from cv_bridge import CvBridge
import rospy

from sensor_msgs.msg import CompressedImage, Image, LaserScan  # do NOT remove, used in eval() expression
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, String

from std_srvs.srv import Empty as Emptyservice

bridge = CvBridge()


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._ros = RosWrapper(
            config=config.ros_config.ros_launch_config.__dict__,
            launch_file='load_ros.launch',
            visible=config.ros_config.visible_xterm
        )
        # Fields
        self._terminal_state = TerminalType.NotDone
        self._sensor_values = {}
        self._step = 0

        # Services
        if config.ros_config.ros_launch_config.gazebo:
            self._pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)

        # Subscribers
        rospy.Subscriber(rospy.get_param('terminal_topic'),
                         String,
                         self._set_terminal_state)

        sensor_list = rospy.get_param('sensors')
        for sensor in sensor_list:
            sensor_name = sensor
            sensor_topic = rospy.get_param(f'{sensor}_topic')
            sensor_type = rospy.get_param(f'{sensor}_type')
            sensor_callback = f'_set_{camelcase_to_snake_format(sensor_type)}'
            sensor_args = rospy.get_param(f'{sensor}_stats') if rospy.has_param(f'{sensor}_stats') else {}
            assert sensor_callback in self.__dir__(), f'Could not find self._set_{sensor} in {self.__file__}'
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=eval(f'self._set_{sensor}'),
                             callback_args=(sensor_name, sensor_args))
            self._sensor_values[sensor_name] = None

        # Publishers
        self._action_pub = rospy.Publisher(rospy.get_param('command_topic'),
                                           Twist, queue_size=1)

    def _set_compressed_image(self, msg, sensor_name: str, sensor_stats: dict) -> None:
        self._sensor_values[sensor_name] = process_compressed_image(msg, sensor_stats)

    def _set_image(self, msg, sensor_name: str, sensor_stats: dict) -> None:
        self._sensor_values[sensor_name] = process_image(msg, sensor_stats)

    def _set_laser_scan(self, msg, sensor_name: str, sensor_stats: dict) -> None:
        self._sensor_values[sensor_name] = process_laser_scan(msg, sensor_stats)

    def _set_terminal_state(self, terminal_msg=None) -> None:
        terminal_translator = {
            'success': TerminalType.Success,
            'failure': TerminalType.Failure
        }
        if terminal_msg is not None:
            self._terminal_state = terminal_translator[terminal_msg]
        else:
            self._terminal_state = TerminalType.NotDone if self._step < self._config.max_number_of_steps \
                else TerminalType.Success

    def reset(self) -> State:
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_physics_client()
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                # TODO make await with asyncio to multitask awaiting different sensors.
                sensor_name: self._await_new_data(sensor_name) for sensor_name in self._sensor_values.keys()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_physics_client()
        return state

    def step(self, action: Action) -> State:
        self._action_pub.publish(adapt_action_to_twist(action))
        self._step += 1
        self._set_terminal_state()
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_physics_client()
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                sensor_name: sensor_value for sensor_name, sensor_value in self._sensor_values.items()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_physics_client()
        return state

    def get_actor(self) -> Actor:
        pass

    def _await_new_data(self, sensor_name: str) -> np.ndarray:
        self._sensor_values[sensor_name] = None
        while self._sensor_values[sensor_name] is None:
            time.sleep(1)
        return self._sensor_values[sensor_name]



