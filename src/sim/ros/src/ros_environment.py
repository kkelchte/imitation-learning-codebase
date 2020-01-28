#!/usr/bin/python3.7
import warnings
from typing import Tuple

import time
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from imitation_learning_ros_package.msg import RosState, RosSensor, RosAction
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.extra_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.utils import camelcase_to_snake_format
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, State, TerminalType, ActorType
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper, ProcessState
from src.sim.ros.src.utils import adapt_action_to_twist, process_compressed_image, process_image, process_laser_scan, \
    adapt_twist_to_action, adapt_sensor_to_ros_message, get_type_from_topic_and_actor_configs

bridge = CvBridge()


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._pause_period = 1./config.ros_config.step_rate_fps
        roslaunch_arguments = config.ros_config.ros_launch_config.__dict__
        # Add automatically added values according to robot_name, world_name, actor_configs
        if config.ros_config.ros_launch_config.robot_name == 'turtlebot_sim' and \
                config.ros_config.ros_launch_config.gazebo:
            roslaunch_arguments['turtlebot_sim'] = True
        for actor_config in config.actor_configs:
            roslaunch_arguments[actor_config.name] = True
            roslaunch_arguments[f'{actor_config.name}_config_file_path_with_extension'] = actor_config.file

        self._ros = RosWrapper(
            config=roslaunch_arguments,
            launch_file='load_ros.launch',
            visible=config.ros_config.visible_xterm
        )
        # Fields
        self._step = 0

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)

        # Subscribers
        if rospy.has_param('/fsm/state_topic'):
            rospy.Subscriber(rospy.get_param('/fsm/state_topic'),
                             String,
                             self._set_field,
                             callback_args='_fsm_state')
        self._fsm_state = FsmState.Unknown

        if rospy.has_param('/fsm/terminal_topic'):
            rospy.Subscriber(rospy.get_param('/fsm/terminal_topic'),
                             String,
                             self._set_field,
                             callback_args='_terminal_state')
        self._terminal_state = TerminalType.Unknown

        sensor_list = rospy.get_param('/robot/sensors', [])
        self._sensor_values = {}
        for sensor in sensor_list:
            sensor_name = sensor
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            sensor_callback = f'_set_{camelcase_to_snake_format(sensor_type)}'
            # if not hasattr(self, sensor_callback):
            #     continue
            assert sensor_callback in self.__dir__(), f'Could not find {sensor_callback} in {self.__module__}'
            sensor_args = rospy.get_param(f'/robot/{sensor}_stats') if rospy.has_param(f'/robot/{sensor}_stats') else {}
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=eval(f'self.{sensor_callback}'),
                             callback_args=(sensor_name, sensor_args))
            self._sensor_values[sensor_name]: np.ndarray = np.zeros((1,))

        self._actor_values = {}
        if self._config.actor_configs is None:
            self._config.actor_configs = []
        if rospy.has_param('/robot/command_topic'):
            self._config.actor_configs.append(
                ActorConfig(
                    name='applied_action',
                    type=ActorType.Unknown,
                    specs={
                        'command_topic': rospy.get_param('/robot/command_topic')
                    }
                )
            )
        if rospy.has_param('/control_mapping/supervision_topic'):
            self._config.actor_configs.append(
                ActorConfig(
                    name='supervised_action',
                    type=ActorType.Unknown,
                    specs={
                        'command_topic': rospy.get_param('/control_mapping/supervision_topic')
                    }
                )
            )
        self.control_mapping = rospy.get_param('/control_mapping/mapping', {})
        for actor_config in self._config.actor_configs:
            self._actor_values[actor_config.name]: Action = Action()
            rospy.Subscriber(name=f'{actor_config.specs["command_topic"]}',
                             data_class=Twist,
                             callback=self._set_action,
                             callback_args=actor_config)
        # Publishers
        # Actions are taken directly by control_mapping node from various actor-topics
        # to avoid slow data saving action in the main process
        # This does mean that control is published to robot at a faster rate than it is stored.
        # self._action_pub = rospy.Publisher(rospy.get_param('command_topic'),
        #                                    Twist, queue_size=1)
        # Publish state at each step
        self._state_publisher = rospy.Publisher(name='/ros_environment/state',
                                                data_class=RosState,
                                                queue_size=10)

        # Start ROS node:
        rospy.init_node('ros_python_interface', anonymous=True)

    def _set_action(self, msg: Twist, config: ActorConfig) -> None:
        action = Action(actor_name=config.name,
                        actor_type=config.type,
                        value=adapt_twist_to_action(msg).value)
        if config.type == ActorType.Unknown and self.control_mapping != {}:
            original_topic = self.control_mapping[self._fsm_state.name]
            action.actor_type = get_type_from_topic_and_actor_configs(actor_configs=self._config.actor_configs,
                                                                      topic_name=original_topic)
        self._actor_values[config.name] = action

    def _set_compressed_image(self, msg: CompressedImage, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_compressed_image(msg, sensor_stats)

    def _set_image(self, msg: Image, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_image(msg, sensor_stats)

    def _set_laser_scan(self, msg: LaserScan, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_laser_scan(msg, sensor_stats)

    def _set_odometry(self, msg: Odometry, args: Tuple) -> None:
        sensor_name, sensor_stats = args
        self._sensor_values[sensor_name] = process_laser_scan(msg, sensor_stats)

    def _set_field(self, msg: String, field_name: str) -> None:
        translator = {
            '_terminal_state': TerminalType[msg],
            '_fsm_state': FsmState[msg],
        }
        eval(f'self.{field_name} = {translator[field_name]}')

    def _check_max_number_of_steps(self):
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

    def reset(self, dont_wait: bool = False) -> State:
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        # TODO add option to reset gazebo robot
        rospy.sleep(self._pause_period)
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                # TODO make await with asyncio to multitask awaiting different sensors.
                sensor_name: self._sensor_values[sensor_name] for sensor_name in self._sensor_values.keys()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()
        return state

    def step(self, action: Action = None, dont_wait: bool = False) -> State:
        # self._action_pub.publish(adapt_action_to_twist(action))
        self._step += 1
        self._check_max_number_of_steps()
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        state = State(
            terminal=self._terminal_state,
            sensor_data={
                sensor_name: self._await_new_sensor_data(sensor_name) if not dont_wait
                else self._sensor_values[sensor_name] for sensor_name in self._sensor_values.keys()
            },
            actor_data={
                actor_name: actor_value for actor_name, actor_value in self._actor_values.items()
            },
            time_stamp_us=int(rospy.get_time() * 10**6)
        )
        self._publish_state(state)
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()
        return state

    def _publish_state(self, state: State):
        ros_state = RosState()
        ros_state.robot_name = self._config.ros_config.ros_launch_config.robot_name
        ros_state.time_stamp_us = state.time_stamp_us
        ros_state.terminal = state.terminal
        ros_state.actions = [adapt_action_to_twist(state.actor_data[actor_name])
                             for actor_name in state.actor_data.keys()]
        ros_state.sensors = [adapt_sensor_to_ros_message(state.sensor_data[sensor_name], sensor_name)
                             for sensor_name in state.sensor_data.keys()]
        self._state_publisher.publish()

    def _await_new_sensor_data(self, sensor_name: str) -> np.ndarray:
        self._sensor_values[sensor_name] = None
        while self._sensor_values[sensor_name] is None:
            # await asyncio.sleep(0.05)
            time.sleep(0.05)
        return self._sensor_values[sensor_name]

    # def _await_new_action_data(self, action_topic: str) -> Action:
    #     self._actor_values[action_topic] = None
    #     while self._actor_values[action_topic] is None:
    #         # await asyncio.sleep(0.05)
    #         time.sleep(0.05)
    #     return self._actor_values[action_topic]

    def remove(self) -> ProcessState:
        return self._ros.terminate()

    def get_actor(self) -> ActorType:
        if self.control_mapping is not {}:
            topic_name = self.control_mapping[self._fsm_state.name]
            return get_type_from_topic_and_actor_configs(self._config.actor_configs, topic_name)


