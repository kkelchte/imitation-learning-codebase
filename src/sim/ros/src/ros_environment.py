#!/usr/bin/python3.8
import os
import signal
import sys
import time
from copy import deepcopy
from typing import Tuple, Union, List, Iterable

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32MultiArray, Empty, Float32
from std_srvs.srv import Empty as Emptyservice, EmptyRequest

from imitation_learning_ros_package.msg import RosReward
from src.core.logger import cprint, MessageType
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.utils import camelcase_to_snake_format
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.actors import ActorConfig
from src.core.data_types import Action, Experience, TerminationType, ProcessState
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import adapt_twist_to_action, quaternion_from_euler, adapt_action_to_twist, process_imu, \
    process_compressed_image, process_image, process_odometry, process_laser_scan

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

        if config.ros_config.actor_configs is not None:
            for actor_config in config.ros_config.actor_configs:
                roslaunch_arguments[actor_config.name] = True
                config_file = actor_config.file if actor_config.file.startswith('/') \
                    else os.path.join(os.environ['HOME'], actor_config.file)
                roslaunch_arguments[f'{actor_config.name}_config_file_path_with_extension'] = config_file

        assert os.path.isfile(os.path.join(os.environ["PWD"], 'src/sim/ros/config/world/',
                                           roslaunch_arguments['world_name']) + '.yml')
        self._ros = RosWrapper(
            config=roslaunch_arguments,
            launch_file='load_ros.launch',
            visible=config.ros_config.visible_xterm
        )

        # Fields
        self._step = 0
        self._current_experience = None
        self._previous_observation = None
        self._info = {}

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
            self._reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Emptyservice)
            self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Subscribers
        # FSM
        self.fsm_state = None
        rospy.Subscriber(name=rospy.get_param('/fsm/state_topic'),
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))

        # Reward topic
        self._reward = None
        self._terminal_state = TerminationType.Unknown
        rospy.Subscriber(name=rospy.get_param('/fsm/reward_topic', ''),
                         data_class=RosReward,
                         callback=self._set_reward)

        # Subscribe to observation
        self._observation = None
        if self._config.ros_config.observation != '':
            self._subscribe_observation(self._config.ros_config.observation)

        # Subscribe to action
        self._action = None
        if self._config.ros_config.action == '' or not self._subscribe_action(self._config.ros_config.action):
            if rospy.has_param('/robot/command_topic'):
                rospy.Subscriber(name=rospy.get_param('/robot/command_topic'),
                                 data_class=Twist,
                                 callback=self._set_field,
                                 callback_args=('action', {}))

        # Add info sensors
        if self._config.ros_config.info is not None:
            self._sensor_processors = {}
            self._subscribe_info(self._config.ros_config.info)

        # Publishers
        self._reset_publisher = rospy.Publisher(rospy.get_param('/fsm/reset_topic', '/reset'), Empty, queue_size=10)
        self._action_publisher = rospy.Publisher('/ros_python_interface/cmd_vel', Twist, queue_size=10)
        # Catch kill signals:
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start ROS node:
        rospy.init_node('ros_python_interface', anonymous=True)

        # assert fsm has started properly
        if self._config.ros_config.ros_launch_config.fsm:
            cprint('Wait till fsm has started properly.', self._logger)
            while self.fsm_state is None:
                self._run_shortly()

        # assert all experience fields are updated once:
        if self._config.ros_config.observation != '':
            cprint('Wait till observation sensor has started properly.', self._logger)
            while self._observation is None:
                self._run_shortly()

        cprint('ready', self._logger)

    def _subscribe_action(self, actor: str) -> bool:
        actor_config = [c for c in self._config.ros_config.actor_configs if c.name == actor]
        if len(actor_config) == 0:
            return False
        rospy.Subscriber(name=f'{actor_config[0].specs["command_topic"]}',
                         data_class=Twist,
                         callback=self._set_field,
                         callback_args=('action', {}))
        return True

    def _subscribe_observation(self, sensor: str) -> None:
        available_sensor_list = rospy.get_param('/robot/sensors', [])
        if sensor.startswith('sensor/'):
            sensor = sensor[7:]
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
            sensor_topic = rospy.get_param(f'/robot/{sensor}_topic')
            sensor_type = rospy.get_param(f'/robot/{sensor}_type')
            try:  # if processing function exist for this sensor name add it to the sensor processing functions
                self._sensor_processors[sensor] = eval(f'process_{camelcase_to_snake_format(sensor_type)}')
            except NameError:
                pass
            else:
                sensor_args = rospy.get_param(f'/robot/{sensor}_stats') \
                    if rospy.has_param(f'/robot/{sensor}_stats') else {}
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=self._set_info_data,
                                 callback_args=(sensor, sensor_args))
                self._info[sensor]: np.ndarray = None

        if 'current_waypoint' in info_objects:
            # add waypoint subscriber as extra sensor
            rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                             data_class=Float32MultiArray,
                             callback=self._set_info_data,
                             callback_args=('current_waypoint', {}))
            self._info['current_waypoint'] = None

        actor_name_list = [info_obj[6:] for info_obj in info_objects if info_obj.startswith('actor/')]
        actor_configs = [config for config in self._config.ros_config.actor_configs if config.name in actor_name_list] \
            if self._config.ros_config.actor_configs is not None else []

        if 'supervised_action' in info_objects and rospy.has_param('/control_mapping/supervision_topic'):
            actor_configs.append(
                ActorConfig(
                    name='supervised_action',
                    specs={
                        'command_topic': rospy.get_param('/control_mapping/supervision_topic')
                    }
                )
            )
        if 'command' in info_objects:
            actor_configs.append(
                ActorConfig(
                    name='applied_action',
                    specs={
                        'command_topic': rospy.get_param('/robot/command_topic', '')
                    }
                )
            )

        for actor_config in actor_configs:
            self._info[actor_config.name] = None
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
            self._info[info_object_name] = self._sensor_processors[info_object_name](msg, extra_args)
        elif isinstance(extra_args, ActorConfig):
            self._info[extra_args.name] = Action(actor_name=extra_args.name,
                                                 value=adapt_twist_to_action(msg).value)
        else:
            self._info[info_object_name] = np.asarray(msg.data)
        cprint(f'set info {info_object_name}', self._logger, msg_type=MessageType.debug)

    def _set_field(self, msg: Union[String, Twist, Image, CompressedImage], args: Tuple) -> None:
        field_name, sensor_stats = args
        if field_name == 'fsm_state':
            self.fsm_state = FsmState[msg.data]
        elif field_name == 'observation':
            self._observation = self._process_observation(msg, sensor_stats)
        elif field_name == 'action':
            self._action = Action(actor_name='applied_action',
                                  value=adapt_twist_to_action(msg).value)
        else:
            raise NotImplementedError
        if not field_name == "observation":  # observation is ros node which keeps on running during gazebo pause.
            cprint(f'set field {field_name}',
                   self._logger,
                   msg_type=MessageType.debug)

    def _set_reward(self, msg: RosReward) -> None:
        self._reward = msg.reward
        self._terminal_state = TerminationType[msg.termination]

    def _internal_update_terminal_state(self):
        if self.fsm_state == FsmState.Running and \
                self._config.max_number_of_steps != -1 and \
                self._config.max_number_of_steps <= self._step:
            self._terminal_state = TerminationType.Done
            cprint(f'reach max number of steps {self._config.max_number_of_steps} < {self._step}', self._logger)

    def _pause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        self._pause_client.wait_for_service()
        self._pause_client(EmptyRequest())
        #os.system("rosservice call gazebo/pause_physics")

    def _unpause_gazebo(self):
        assert self._config.ros_config.ros_launch_config.gazebo
        self._unpause_client.wait_for_service()
        self._unpause_client(EmptyRequest())
        #os.system("rosservice call gazebo/unpause_physics")

    def _reset_gazebo(self):
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
        #os.system(f"rosservice call /gazebo/set_model_state '{{model_state: "
        #          f"{{ model_name: {model_state.model_name},"
        #          f"pose: {{ position: {{ x: {model_state.pose.position.x},"
        #          f"                      y: {model_state.pose.position.y},"
        #          f"                      z: {model_state.pose.position.z}, }},"
        #          f"         orientation: {{ x: { model_state.pose.orientation.x},"
        #          f"                         y: { model_state.pose.orientation.y},"
        #          f"                         z: { model_state.pose.orientation.z},"
        #          f"                         w: { model_state.pose.orientation.w},}} }} }} }}'")

    def _clear_experience_values(self):
        """Set all experience fields to None"""
        self._observation = None
        self._action = None
        self._reward = None
        self._terminal_state = None
        self._info = {k: None for k in self._info.keys()}

    def _update_current_experience(self) -> bool:
        """
        If all experience fields are updated,
        store all experience fields in _current_experience fields end return True
        else False.
        :return: Bool whether all fields are updated
        """
        self._internal_update_terminal_state()
        if self._config.ros_config.observation != '' and self._observation is None:
            cprint("waiting for observation", self._logger, msg_type=MessageType.debug)
            return False
        if self._terminal_state is None:
            cprint("waiting for terminal state", self._logger, msg_type=MessageType.debug)
            return False
        if self._config.ros_config.store_reward and self._reward is None:
            cprint("waiting for reward", self._logger, msg_type=MessageType.debug)
            return False
        if self._config.ros_config.store_action and self._action is None:
            cprint("waiting for action", self._logger, msg_type=MessageType.debug)
            return False
        if None in [v for v in self._info.values() if not isinstance(v, Iterable)]:
            cprint("waiting for info", self._logger, msg_type=MessageType.debug)
            return False
        self._current_experience = Experience(
            done=deepcopy(self._terminal_state),
            observation=deepcopy(self._previous_observation
                                 if self._previous_observation is not None else self._observation),
            action=deepcopy(self._action) if self._config.ros_config.store_action else None,
            reward=deepcopy(self._reward) if self._config.ros_config.store_reward else None,
            time_stamp=int(rospy.get_time() * 10 ** 3),
            info={
                field_name: deepcopy(self._info[field_name]) for field_name in self._info.keys()
            }
        )
        cprint(f"update current experience: "
               f"done {self._current_experience.done}, "
               f"reward {self._current_experience.reward}, "
               f"time_stamp {self._current_experience.time_stamp}, "
               f"info: {[k for k in self._current_experience.info.keys()]}", self._logger, msg_type=MessageType.debug)
        self._previous_observation = deepcopy(self._observation)
        return True

    def _run_shortly(self):
        if self._config.ros_config.ros_launch_config.gazebo:
            self._unpause_gazebo()
        rospy.sleep(self._pause_period)
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_gazebo()

    def _run_and_update_experience(self):
        self._clear_experience_values()
        start_rospy_time = rospy.get_time()
        start_time = time.time()
        self._run_shortly()
        while not self._update_current_experience():
            self._run_shortly()
            if time.time() - start_time > self._config.ros_config.max_update_wait_period_s:
                cprint(f"ros seems to be stuck, waiting for more than "
                       f"{self._config.ros_config.max_update_wait_period_s}s, so exit.",
                       self._logger,
                       msg_type=MessageType.warning)
                self.remove()
                sys.exit(1)
        self._current_experience.info['run_time_duration_s'] = rospy.get_time() - start_rospy_time

    def reset(self) -> Tuple[Experience, np.ndarray]:
        """
        reset gazebo, reset fsm, wait till fsm in 'running' state
        return experience without reward or action
        """
        cprint(f'resetting', self._logger)
        self._step = 0
        self._reset_publisher.publish(Empty())
        if self._config.ros_config.ros_launch_config.gazebo:
            self._reset_gazebo()
        self._clear_experience_values()
        while self.fsm_state != FsmState.Running \
                or self._observation is None \
                or self._terminal_state is None:
            self._run_shortly()
        self._current_experience = Experience(
            done=deepcopy(self._terminal_state),
            observation=deepcopy(self._observation),
            time_stamp=int(rospy.get_time() * 10 ** 3),
        )
        self._previous_observation = deepcopy(self._observation)
        return self._current_experience, deepcopy(self._observation)

    def step(self, action: Action = None) -> Tuple[Experience, np.ndarray]:
        self._step += 1
        if action is not None:
            self._action_publisher.publish(adapt_action_to_twist(action))
        self._run_and_update_experience()
        return self._current_experience, deepcopy(self._observation)

    def remove(self) -> bool:
        return self._ros.terminate() == ProcessState.Terminated
