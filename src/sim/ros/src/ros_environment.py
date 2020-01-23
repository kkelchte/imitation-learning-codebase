#!/usr/bin/python3.7
import warnings
from typing import Tuple

import time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from src.sim.ros.extra_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.utils import camelcase_to_snake_format
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, State, TerminalType
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import adapt_action_to_twist, process_compressed_image, process_image, process_laser_scan, \
    adapt_twist_to_action

bridge = CvBridge()


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        # TODO: for each actor config with specs:
        # TODO:     > add node in ros_launch config automatically with file path to actor config as argument:
        #           expert_config_file_path_with_extension
        #           (!) complain if no config.file is provided or when specs != None

        super().__init__(config)
        self._ros = RosWrapper(
            config=config.ros_config.ros_launch_config.__dict__,
            launch_file='load_ros.launch',
            visible=config.ros_config.visible_xterm
        )
        # Fields
        self._terminal_state = TerminalType.Unknown
        self._sensor_values = {}
        self._actor_values = {}
        self._step = 0

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)

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
            if not hasattr(self, sensor_callback):
                continue
            # assert sensor_callback in self.__dir__(), f'Could not find {sensor_callback} in {self.__module__}'
            sensor_args = rospy.get_param(f'{sensor}_stats') if rospy.has_param(f'{sensor}_stats') else {}
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=eval(f'self.{sensor_callback}'),
                             callback_args=(sensor_name, sensor_args))
            self._sensor_values[sensor_name]: np.ndarray = np.zeros((1,))
            # TODO: save sensor_stats to config

        if self._config.actor_configs is not None:
            for actor_config in self._config.actor_configs:
                self._actor_values[actor_config.name]: Action = Action()
                rospy.Subscriber(name=f'{actor_config.name.specs["command_topic"]}',
                                 data_class=Twist,
                                 callback=self._set_action,
                                 callback_args=actor_config)
        # Publishers
        # Actions are taken directly by control_mapping node from various actor-topics
        # to avoid slow data saving action in the main process
        # This does mean that control is published to robot at a faster rate than it is stored.
        # self._action_pub = rospy.Publisher(rospy.get_param('command_topic'),
        #                                    Twist, queue_size=1)

        # Start ROS node:
        rospy.init_node('ros_python_interface', anonymous=True)

    def _set_action(self, msg: Twist, args: Tuple) -> None:
        action = Action(actor_name=args[0],
                        actor_type=args[1],
                        value=adapt_twist_to_action(msg).value
                        )
        self._actor_values[args[0]] = action

    def _set_compressed_image(self, msg: CompressedImage, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_compressed_image(msg, sensor_stats)

    def _set_image(self, msg: Image, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_image(msg, sensor_stats)

    def _set_laser_scan(self, msg: LaserScan, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_laser_scan(msg, sensor_stats)

    def _set_terminal_state(self, terminal_msg=None) -> None:
        if terminal_msg is not None:
            self._terminal_state = TerminalType(terminal_msg)
        else:
            self._terminal_state = TerminalType.NotDone if self._step < self._config.max_number_of_steps \
                else TerminalType.Success

    def _pause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            self._pause_client.call(EmptyRequest())

    def _unpause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            self._unpause_client.call(EmptyRequest())

    def reset(self) -> State:
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                # TODO make await with asyncio to multitask awaiting different sensors.
                sensor_name: self._await_new_data(sensor_name)
                for sensor_name in self._sensor_values.keys()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()
        return state

    def step(self, action: Action) -> State:
        self._action_pub.publish(adapt_action_to_twist(action))
        self._step += 1
        self._set_terminal_state()
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                sensor_name: sensor_value for sensor_name, sensor_value in self._sensor_values.items()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()
        return state

    def _await_new_data(self, sensor_name: str) -> np.ndarray:
        self._sensor_values[sensor_name] = None
        while self._sensor_values[sensor_name] is None:
            # await asyncio.sleep(0.05)
            time.sleep(0.05)
        return self._sensor_values[sensor_name]

    def remove(self):
        self._ros.terminate()


