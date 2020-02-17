#!/usr/bin/python3.7
import signal
import sys
import time
from typing import Tuple, Union

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32MultiArray, Empty
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from imitation_learning_ros_package.msg import RosState

from src.core.logger import cprint, MessageType
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.utils import camelcase_to_snake_format
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, State, TerminalType, ActorType, ProcessState
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import process_compressed_image, process_image, process_laser_scan, \
    adapt_twist_to_action, get_type_from_topic_and_actor_configs, process_odometry, \
    adapt_action_to_ros_message, adapt_action_to_twist, adapt_sensor_to_ros_message, quaternion_from_euler

bridge = CvBridge()


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._pause_period = 1./config.ros_config.step_rate_fps
        roslaunch_arguments = config.ros_config.ros_launch_config.__dict__
        # Add automatically added values according to robot_name, world_name, actor_configs
        if config.ros_config.ros_launch_config.gazebo:
            roslaunch_arguments['turtlebot_sim'] = True \
                if config.ros_config.ros_launch_config.robot_name == 'turtlebot_sim' else False
            roslaunch_arguments['drone_sim'] = True \
                if config.ros_config.ros_launch_config.robot_name == 'drone_sim' else False

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
        self._state = State(
            terminal=TerminalType.Unknown,
            actor_data={},
            sensor_data={},
            time_stamp_ms=-1
        )
        self._default_sensor_value = np.zeros((1,))

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
            self._reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Emptyservice)
            self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Subscribers
        self.fsm_state = None
        if self._config.ros_config.ros_launch_config.fsm:
            # if rospy.has_param('/fsm/state_topic'):
            rospy.Subscriber(rospy.get_param('/fsm/state_topic'),
                             String,
                             self._set_field,
                             callback_args='fsm_state')

        if rospy.has_param('/fsm/terminal_topic'):
            rospy.Subscriber(rospy.get_param('/fsm/terminal_topic'),
                             String,
                             self._set_field,
                             callback_args='_terminal_state')
        self._terminal_state = TerminalType.Unknown

        sensor_list = rospy.get_param('/robot/sensors', [])
        self._sensor_processors = {}
        self._sensor_values = {}
        for sensor in sensor_list:
            sensor_name = sensor
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            try:  # if processing function exist for this sensor name add it to the sensor processing functions
                processing_function = f'process_{camelcase_to_snake_format(sensor_type)}'
                self._sensor_processors[sensor_name] = eval(processing_function)
            except NameError:
                pass
            # sensor_callback = f'_set_{camelcase_to_snake_format(sensor_type)}'
            # assert sensor_callback in self.__dir__(), f'Could not find {sensor_callback} in {self.__module__}'
            sensor_args = rospy.get_param(f'/robot/{sensor}_stats') if rospy.has_param(f'/robot/{sensor}_stats') else {}
            rospy.Subscriber(name=sensor_topic,
                             data_class=eval(sensor_type),
                             callback=self._set_sensor_data,
                             callback_args=(sensor_name, sensor_args))
            self._sensor_values[sensor_name]: np.ndarray = self._default_sensor_value
        # add waypoint subscriber as extra sensor
        sensor_name = 'current_waypoint'
        rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                         data_class=Float32MultiArray,
                         callback=self._set_sensor_data,
                         callback_args=(sensor_name, {}))
        self._sensor_values[sensor_name] = self._default_sensor_value

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
        # Publish state at each step (all except RGB info)
        self._state_publisher = rospy.Publisher('/ros_environment/state', RosState, queue_size=10)
        self._reset_publisher = rospy.Publisher(rospy.get_param('/fsm/reset_topic', '/reset'), Empty, queue_size=10)

        # Catch kill signals:
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start ROS node:
        rospy.init_node('ros_python_interface', anonymous=True)

    def _signal_handler(self, signal_number: int, _) -> None:
        return_value = self.remove()
        cprint(f'received signal {signal_number}.', self._logger,
               msg_type=MessageType.info if return_value == ProcessState.Terminated else MessageType.error)
        sys.exit(0)

    def _set_action(self, msg: Twist, config: ActorConfig) -> None:
        action = Action(actor_name=config.name,
                        actor_type=config.type,
                        value=adapt_twist_to_action(msg).value)
        if config.type == ActorType.Unknown and self.control_mapping != {}:
            original_topic = self.control_mapping[self.fsm_state.name]
            action.actor_type = get_type_from_topic_and_actor_configs(actor_configs=self._config.actor_configs,
                                                                      topic_name=original_topic)
        self._actor_values[config.name] = action

    def _set_sensor_data(self, msg: Union[CompressedImage, Image, LaserScan, Odometry, Float32MultiArray],
                         args: Tuple) -> None:
        sensor_name, sensor_stats = args
        if sensor_name in self._sensor_processors.keys():
            self._sensor_values[sensor_name] = self._sensor_processors[sensor_name](msg, sensor_stats)
        else:
            self._sensor_values[sensor_name] = np.asarray(msg.data)

    def _set_field(self, msg: Union[String, ], field_name: str) -> None:
        if field_name == '_terminal_state':
            self._terminal_state = TerminalType[msg.data]
        elif field_name == 'fsm_state':
            self.fsm_state = FsmState[msg.data]
        else:
            raise NotImplementedError
        cprint(f'received: {msg} and set field {field_name}',
               self._logger,
               msg_type=MessageType.debug)

    def _internal_update_terminal_state(self):
        if self.fsm_state == FsmState.Running and self._terminal_state == TerminalType.Unknown:
            self._terminal_state = TerminalType.NotDone
        if self._terminal_state == TerminalType.NotDone and \
                self._config.max_number_of_steps != -1 and \
                self._config.max_number_of_steps < self._step:
            self._terminal_state = TerminalType.Failure

    def _pause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        try:
            self._pause_client.call(EmptyRequest())
        except ResourceWarning:
            pass

    def _unpause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        try:
            self._unpause_client.wait_for_service()
            self._unpause_client.call(EmptyRequest())
        except ResourceWarning:
            pass

    def _reset_gazebo(self):
        try:
            model_state = ModelState()
            model_state.model_name = 'turtlebot3_burger' \
                if self._config.ros_config.ros_launch_config.robot_name.startswith('turtle') else 'quadrotor'
            model_state.pose = Pose()
            model_state.pose.position.x = self._config.ros_config.ros_launch_config.x_pos
            model_state.pose.position.y = self._config.ros_config.ros_launch_config.y_pos
            model_state.pose.position.z = self._config.ros_config.ros_launch_config.z_pos
            model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, \
                model_state.pose.orientation.w = quaternion_from_euler((0, 0,
                                                                        self._config.ros_config.ros_launch_config.yaw_or))
            self._set_model_state.wait_for_service()
            self._set_model_state(model_state)
        except ResourceWarning:
            pass

    def _run_shortly(self):
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        rospy.sleep(self._pause_period)
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()

    def reset(self) -> State:
        if self._config.ros_config.ros_launch_config.fsm:
            while self.fsm_state is None:
                self._run_shortly()
        self._reset_publisher.publish(Empty())
        if self._config.ros_config.ros_launch_config.gazebo:
            self._reset_gazebo()
        self._run_shortly()
        self._state = State(
            terminal=self._terminal_state,
            sensor_data={
                sensor_name: self._sensor_values[sensor_name] for sensor_name in self._sensor_values.keys()
            },
            actor_data={
                actor_name: actor_value for actor_name, actor_value in self._actor_values.items()
            },
            time_stamp_ms=int(rospy.get_time() * 10**6)
        )
        return self._state

    def _clear_sensor_and_actor_values(self):
        self._sensor_values = {
            sensor_name: self._default_sensor_value for sensor_name in self._sensor_values.keys()
        }
        self._actor_values = {
            actor_name: Action() for actor_name in self._actor_values.keys()
        }

    def step(self, action: Action = None) -> State:
        self._step += 1
        self._internal_update_terminal_state()
        self._clear_sensor_and_actor_values()
        self._run_shortly()
        assert not (self.fsm_state == FsmState.Terminated and self._terminal_state == TerminalType.Unknown)
        self._state = State(
            terminal=self._terminal_state,
            sensor_data={
                sensor_name: self._sensor_values[sensor_name] for sensor_name in self._sensor_values.keys()
                if self._sensor_values[sensor_name].shape != self._default_sensor_value.shape
            },
            actor_data={
                actor_name: actor_value for actor_name, actor_value in self._actor_values.items()
            },
            time_stamp_ms=int(rospy.get_time() * 10**3)
        )
        self._publish_state()
        self._terminal_state = TerminalType.Unknown
        return self._state

    def _publish_state(self):
        ros_state = RosState()
        ros_state.robot_name = self._config.ros_config.ros_launch_config.robot_name
        ros_state.time_stamp_ms = int(self._state.time_stamp_ms)
        ros_state.terminal = self._state.terminal.name
        ros_state.actions = [adapt_action_to_ros_message(self._state.actor_data[actor_name])
                             for actor_name in self._state.actor_data.keys()
                             if adapt_action_to_twist(self._state.actor_data[actor_name]) is not None]
        ros_state.sensors = [adapt_sensor_to_ros_message(self._state.sensor_data[sensor_name], sensor_name=sensor_name)
                             for sensor_name in self._state.sensor_data.keys()
                             if self._state.sensor_data[sensor_name].shape != self._default_sensor_value.shape
                             and 'camera' not in sensor_name and 'image' not in sensor_name
                             and 'depth' not in sensor_name and 'scan' not in sensor_name]
        self._state_publisher.publish(ros_state)

    def remove(self) -> bool:
        return self._ros.terminate() == ProcessState.Terminated

    def get_actor(self) -> ActorType:
        if self.control_mapping is not {}:
            topic_name = self.control_mapping[self.fsm_state.name]
            return get_type_from_topic_and_actor_configs(self._config.actor_configs, topic_name)


