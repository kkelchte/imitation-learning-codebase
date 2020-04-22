#!/usr/bin/python3.7
import os
import signal
import sys
from typing import Tuple, Union, List

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, LaserScan, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32MultiArray, Empty
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from imitation_learning_ros_package.msg import RosState

from src.core.logger import cprint, MessageType
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.utils import camelcase_to_snake_format
from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, Experience, TerminationType, ProcessState
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import process_compressed_image, process_image, process_laser_scan, \
    adapt_twist_to_action, get_type_from_topic_and_actor_configs, process_odometry, process_imu, \
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

        for actor_config in config.ros_config.actor_configs:
            roslaunch_arguments[actor_config.name] = True
            config_file = actor_config.file if actor_config.file.startswith('/') \
                else os.path.join(os.environ['HOME'], actor_config.file)
            roslaunch_arguments[f'{actor_config.name}_config_file_path_with_extension'] = config_file

        self._ros = RosWrapper(
            config=roslaunch_arguments,
            launch_file='load_ros.launch',
            visible=config.ros_config.visible_xterm
        )

        # Fields
        self._step = 0
        self._current_experience = None
        self._default_observation = np.zeros((100, 100, 3), dtype=np.uint8)
        self._default_action = np.ones((1,), dtype=np.float32) * 999
        self._default_reward = np.ones((1,), dtype=np.float32) * 999
        self._default_sensor_value = np.ones((1,), dtype=np.float32) * 999
        self._info = {}
        self._clear_experience_values()

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
            self._reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Emptyservice)
            self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Subscribers
        # FSM
        self.fsm_state = None
        self._terminal_state = TerminationType.Unknown
        rospy.Subscriber(name=rospy.get_param('/fsm/state_topic'),
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))
        # Terminal topic
        rospy.Subscriber(name=rospy.get_param('/fsm/terminal_topic', ''),
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('_terminal_state', {}))

        # Subscribe to observation
        self._observation = self._default_observation
        self._subscribe_observation(self._config.ros_config.observation_sensor)

        # Subscribe to action
        self._action = self._default_action
        rospy.Subscriber(name=rospy.get_param('/robot/command_topic', ''),
                         data_class=Twist,
                         callback=self._set_field,
                         callback_args=('_action', {}))

        # Add info sensors
        if self._config.ros_config.info_sensors is not None:
            self._sensor_processors = {}
            self._info_values = {}
            self._subscribe_info(self._config.ros_config.info_sensors)

        # Publishers
        # Publish state at each step (all except RGB info)
        self._state_publisher = rospy.Publisher('/ros_environment/state', RosState, queue_size=10)
        self._reset_publisher = rospy.Publisher(rospy.get_param('/fsm/reset_topic', '/reset'), Empty, queue_size=10)

        # Catch kill signals:
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start ROS node:
        rospy.init_node('ros_python_interface', anonymous=True)

        # assert fsm has started properly
        if self._config.ros_config.ros_launch_config.fsm:
            cprint('Wait till fsm has started properly.', self._logger)
            while self.fsm_state is None:
                self._run_shortly()

        # assert dnn model has started properly
        if 'dnn_actor' in [conf.name for conf in self._config.actor_configs]:
            cprint('Wait till dnn_actor has started properly.', self._logger)
            while self._action == self._default_action:
                self._run_shortly()

    def _subscribe_observation(self, sensor: str) -> None:
        available_sensor_list = rospy.get_param('/robot/sensors', [])
        assert sensor in available_sensor_list
        sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
        sensor_type = rospy.get_param(f'/robot/{sensor}_type')
        sensor_args = rospy.get_param(f'/robot/{sensor}_stats') if rospy.has_param(f'/robot/{sensor}_stats') else {}
        rospy.Subscriber(name=sensor_topic,
                         data_class=eval(sensor_type),
                         callback=self._set_field,
                         callback_args=('observation', sensor_args))
        self._process_observation = eval(f'process_{camelcase_to_snake_format(sensor_type)}')

    def _subscribe_info(self, info_objects: List[str]) -> None:
        sensor_list = [info_obj[7:] for info_obj in info_objects if info_obj.startswith('sensor/')]
        for sensor in sensor_list:
            try:  # if processing function exist for this sensor name add it to the sensor processing functions
                sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
                sensor_type = rospy.get_param(f'/robot/{sensor}_type')
                self._sensor_processors[sensor] = eval(f'process_{camelcase_to_snake_format(sensor_type)}')
            except NameError:
                pass
            else:
                sensor_args = rospy.get_param(f'/robot/{sensor}_stats') if rospy.has_param(
                    f'/robot/{sensor}_stats') else {}
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=self._set_info_data,
                                 callback_args=(sensor, sensor_args))
                self._info_values[sensor]: np.ndarray = self._default_sensor_value

        if 'current_waypoint' in info_objects:
            # add waypoint subscriber as extra sensor
            rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                             data_class=Float32MultiArray,
                             callback=self._set_info_data,
                             callback_args=('current_waypoint', {}))
            self._info_values['current_waypoint'] = self._default_sensor_value

        actor_name_list = [info_obj[6:] for info_obj in info_objects if info_obj.startswith('actor/')]
        actor_configs = [config for config in self._config.actor_configs if config.name in actor_name_list]

        if 'supervised_action' in info_objects and rospy.has_param('/control_mapping/supervision_topic'):
            actor_configs.append(
                ActorConfig(
                    name='supervised_action',
                    specs={
                        'command_topic': rospy.get_param('/control_mapping/supervision_topic')
                    }
                )
            )

        for actor_config in actor_configs:
            self._info_values[actor_config.name] = self._default_action
            rospy.Subscriber(name=f'{actor_config.specs["command_topic"]}',
                             data_class=Twist,
                             callback=self._set_info_data,
                             callback_args=(actor_config.name, actor_config))

    def _signal_handler(self, signal_number: int, _) -> None:
        return_value = self.remove()
        cprint(f'received signal {signal_number}.', self._logger,
               msg_type=MessageType.info if return_value == ProcessState.Terminated else MessageType.error)
        sys.exit(0)

    def _set_info_data(self, msg: Union[CompressedImage, Image, LaserScan, Odometry, Float32MultiArray, Twist],
                       args: Tuple) -> None:
        info_object_name, extra_args = args
        if info_object_name in self._sensor_processors.keys():
            self._info_values[info_object_name] = self._sensor_processors[info_object_name](msg, extra_args)
        elif isinstance(extra_args, ActorConfig):
            self._info_values[extra_args.name] = Action(actor_name=extra_args.name,
                                                        value=adapt_twist_to_action(msg).value)
        else:
            self._info_values[info_object_name] = np.asarray(msg.data)
        cprint(f'set info {info_object_name}', self._logger)

    def _set_field(self, msg: String, args: Tuple) -> None:
        field_name, sensor_stats = args
        if field_name == '_terminal_state':
            self._terminal_state = TerminationType[msg.data]
        elif field_name == 'fsm_state':
            self.fsm_state = FsmState[msg.data]
        elif field_name == 'observation':
            self._observation = self._process_observation(msg, sensor_stats)
        else:
            raise NotImplementedError
        cprint(f'set field {field_name}',
               self._logger,
               msg_type=MessageType.debug)

    def _internal_update_terminal_state(self):
        if self.fsm_state == FsmState.Running and self._terminal_state == TerminationType.Unknown:
            self._terminal_state = TerminationType.NotDone
        if self._terminal_state == TerminationType.NotDone and \
                self._config.max_number_of_steps != -1 and \
                self._config.max_number_of_steps < self._step:
            self._terminal_state = TerminationType.Done
            cprint(f'reach max number of steps {self._config.max_number_of_steps} < {self._step}', self._logger)

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
                model_state.pose.orientation.w = quaternion_from_euler(
                    (0, 0, self._config.ros_config.ros_launch_config.yaw_or))
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

    def reset(self) -> Experience:
        cprint(f'resetting', self._logger)
        self._step = 0
        self._terminal_state = TerminationType.Unknown
        self._reset_publisher.publish(Empty())
        if self._config.ros_config.ros_launch_config.gazebo:
            self._reset_gazebo()
        self._run_shortly()
        self._info['time_stamp_ms'] = int(rospy.get_time() * 10 ** 3)
        self._current_experience = Experience(
            done=self._terminal_state,
            observation=self._observation,
            action=self._action,
            reward=self._reward,
            info={
                field_name: self._info[field_name] for field_name in self._info.keys()
            }
        )
        return self._current_experience

    def _clear_experience_values(self):
        self._observation = self._default_observation
        self._action = self._default_action
        self._reward = self._default_reward
        self._info = {}

    def step(self, action: Action = None) -> Experience:
        self._step += 1
        self._internal_update_terminal_state()
        self._clear_experience_values()
        self._run_shortly()
        assert not (self.fsm_state == FsmState.Terminated and self._terminal_state == TerminationType.Unknown)
        self._info['time_stamp_ms'] = int(rospy.get_time() * 10**3)
        self._current_experience = Experience(
            done=self._terminal_state,
            observation=self._observation,
            action=self._action,
            reward=self._reward,
            info={
                field_name: self._info[field_name] for field_name in self._info.keys()
            }
        )
        self._publish_state()
        self._terminal_state = TerminationType.Unknown
        return self._current_experience

    def _publish_state(self):
        ros_state = RosState()
        ros_state.robot_name = self._config.ros_config.ros_launch_config.robot_name
        ros_state.time_stamp_ms = int(self._current_experience.info['time_stamp_ms'])
        ros_state.terminal = self._current_experience.done.name
        ros_state.action = self._current_experience.action != self._default_action
        ros_state.reward = self._current_experience.reward != self._default_reward
        ros_state.observation = self._current_experience.observation != self._default_observation
        self._state_publisher.publish(ros_state)

    def remove(self) -> bool:
        return self._ros.terminate() == ProcessState.Terminated

    # def get_actor(self) -> ActorType:
    #     if self.control_mapping is not {}:
    #         topic_name = self.control_mapping[self.fsm_state.name]
    #         return get_type_from_topic_and_actor_configs(self._config.actor_configs, topic_name)
