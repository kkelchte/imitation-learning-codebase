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

from imitation_learning_ros_package.msg import RosReward, CombinedGlobalPoses
from src.core.logger import cprint, MessageType
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge.core import CvBridge
from src.core.utils import camelcase_to_snake_format, ros_message_to_type_str
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.actors import ActorConfig
from src.core.data_types import Action, Experience, TerminationType, ProcessState, SensorType
from src.sim.common.environment import EnvironmentConfig, Environment
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.utils import quaternion_from_euler, adapt_action_to_twist, process_imu, \
    process_compressed_image, process_image, process_odometry, process_laser_scan, process_twist, \
    process_float32multi_array, process_pose_stamped, process_combined_global_poses

bridge = CvBridge()


class RosEnvironment(Environment):

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self._pause_period = 1./config.ros_config.step_rate_fps
        roslaunch_arguments = config.ros_config.ros_launch_config.__dict__
        # Add automatically added values according to robot_name, world_name, actor_configs
        # if config.ros_config.ros_launch_config.robot_name is not None:
        #     roslaunch_arguments[config.ros_config.ros_launch_config.robot_name] = True

        if config.ros_config.actor_configs is not None:
            for actor_config in config.ros_config.actor_configs:
                roslaunch_arguments[actor_config.name] = True
                config_file = actor_config.file if actor_config.file.startswith('/') \
                    else os.path.join(os.environ['CODEDIR'], actor_config.file)
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
        self._return = 0
        self._current_experience = None
        self._previous_observation = None

        # Services
        if self._config.ros_config.ros_launch_config.gazebo:
            self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
            self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
            self._reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Emptyservice)
            self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Subscribers
        # FSM
        self.fsm_state = None
        rospy.Subscriber(name='/fsm/state',
                         data_class=String,
                         callback=self._set_field,
                         callback_args=('fsm_state', {}))

        # Reward topic
        self.reward = None
        self.terminal_state = TerminationType.Unknown
        rospy.Subscriber(name='/fsm/reward',
                         data_class=RosReward,
                         callback=self._set_field,
                         callback_args=('reward', {}))

        # Subscribe to observation
        self.observation = None
        if self._config.ros_config.observation != '':
            sensor = SensorType[self._config.ros_config.observation]
            rospy.Subscriber(name=rospy.get_param(f'/robot/{sensor.name}_sensor/topic'),
                             data_class=eval(rospy.get_param(f'/robot/{sensor.name}_sensor/type')),
                             callback=self._set_field,
                             callback_args=('observation', rospy.get_param(f'/robot/{sensor.name}_sensor/stats', {})))

        # Subscribe to action
        self.action = None
        if self._config.ros_config.action_topic != 'python':
            rospy.Subscriber(name=self._config.ros_config.action_topic,
                             data_class=Twist,
                             callback=self._set_field,
                             callback_args=('action', {}))

        # Add info sensors
        self.info = {}
        if self._config.ros_config.info is not None:
            self._subscribe_info()

        # Publishers
        self._reset_publisher = rospy.Publisher('/fsm/reset', Empty, queue_size=10)
        self._action_publishers = [rospy.Publisher(f'/ros_python_interface/cmd_vel{f"_{i}" if i != 0 else ""}',
                                                   Twist, queue_size=10)
                                   for i in range(self._config.ros_config.num_action_publishers)]
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
            while self.observation is None:
                self._run_shortly()
        cprint('ready', self._logger)

    def _set_field(self, msg, args: Tuple) -> None:
        field_name, sensor_stats = args
        if field_name == 'fsm_state':
            self.fsm_state = FsmState[msg.data]
        elif field_name == 'observation':
            msg_type = camelcase_to_snake_format(ros_message_to_type_str(msg))
            self.observation = eval(f'process_{msg_type}(msg, sensor_stats)')
        elif field_name == 'action':
            self.action = Action(actor_name=self._config.ros_config.action_topic,
                                 value=process_twist(msg).value)
        elif field_name == 'reward':
            self.reward = msg.reward
            self.terminal_state = TerminationType[msg.termination]
        elif field_name.startswith('info:'):
            info_key = field_name[5:]
            msg_type = camelcase_to_snake_format(ros_message_to_type_str(msg))
            self.info[info_key] = eval(f'process_{msg_type}(msg, sensor_stats)')
        else:
            raise NotImplementedError(f'{field_name}: {msg}')
        cprint(f'set field {field_name}',
               self._logger,
               msg_type=MessageType.debug)

    def _subscribe_info(self) -> None:
        for info in self._config.ros_config.info:
            if info in [s.name for s in SensorType]:
                sensor_topic = rospy.get_param(f'/robot/{info}_sensor/topic')
                sensor_type = rospy.get_param(f'/robot/{info}_sensor/type')
                sensor_stats = rospy.get_param(f'/robot/{info}_sensor/stats', {})
                rospy.Subscriber(name=sensor_topic,
                                 data_class=eval(sensor_type),
                                 callback=self._set_field,
                                 callback_args=(f'info:{info}', sensor_stats))
            elif info == 'current_waypoint':
                # add waypoint subscriber as extra sensor
                rospy.Subscriber(name='/waypoint_indicator/current_waypoint',
                                 data_class=Float32MultiArray,
                                 callback=self._set_field,
                                 callback_args=('info:current_waypoint', {}))
            elif info.startswith('/'):  # subscribe to rostopic for action, expecting twist message
                rospy.Subscriber(name=info,
                                 data_class=Twist,
                                 callback=self._set_field,
                                 callback_args=(f'info:{info}', {}))
            self.info[info] = None

    def _signal_handler(self, signal_number: int, _) -> None:
        return_value = self.remove()
        cprint(f'received signal {signal_number}.', self._logger,
               msg_type=MessageType.info if return_value == ProcessState.Terminated else MessageType.error)
        sys.exit(0)

    def _internal_update_terminal_state(self):
        if self.fsm_state == FsmState.Running and \
                self._config.max_number_of_steps != -1 and \
                self._config.max_number_of_steps <= self._step:
            self.terminal_state = TerminationType.Done
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
        self.observation = None
        if self._config.ros_config.action_topic != 'python':
            self.action = None
        self.reward = None
        self.terminal_state = None
        self.info = {k: None for k in self.info.keys() if k != 'unfiltered_reward' and k != 'return'}
        if 'return' in self.info.keys():
            del self.info['return']

    def _update_current_experience(self) -> bool:
        """
        If all experience fields are updated,
        store all experience fields in _current_experience fields end return True
        else False.
        :return: Bool whether all fields are updated
        """
        self._internal_update_terminal_state()  # check count_steps for termination
        if self._config.ros_config.observation != '' and self.observation is None:
            cprint("waiting for observation", self._logger, msg_type=MessageType.debug)
            return False
        if self.reward is None:
            cprint("waiting for reward", self._logger, msg_type=MessageType.debug)
            return False
        if self.terminal_state is None:
            cprint("waiting for terminal state", self._logger, msg_type=MessageType.debug)
            return False
        if self.action is None and self.terminal_state == TerminationType.NotDone:
            # Don't wait for next action if episode is finished
            cprint("waiting for action", self._logger, msg_type=MessageType.debug)
            return False
        if None in [v for v in self.info.values() if not isinstance(v, Iterable)] and \
                self.terminal_state == TerminationType.NotDone:  # Don't wait for next info if episode is finished:
            cprint("waiting for info", self._logger, msg_type=MessageType.debug)
            return False
        self.observation = self._filter_observation(self.observation)
        self.info['unfiltered_reward'] = deepcopy(self.reward)
        self._return += self.reward
        self.reward = self._filter_reward(self.reward)
        if self.terminal_state in [TerminationType.Done, TerminationType.Success, TerminationType.Failure]:
            self.info['return'] = self._return

        self._current_experience = Experience(
            done=deepcopy(self.terminal_state),
            observation=deepcopy(self._previous_observation),
            action=deepcopy(self.action),
            reward=deepcopy(self.reward),
            time_stamp=int(rospy.get_time() * 10 ** 3),
            info={
                field_name: deepcopy(self.info[field_name]) for field_name in self.info.keys()
            }
        )
        cprint(f"update current experience: "
               f"done {self._current_experience.done}, "
               f"reward {self._current_experience.reward}, "
               f"time_stamp {self._current_experience.time_stamp}, "
               f"info: {[k for k in self._current_experience.info.keys()]}", self._logger, msg_type=MessageType.debug)
        self._previous_observation = deepcopy(self.observation)
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
        self._reset_filters()
        self._step = 0
        self._return = 0
        self._reset_publisher.publish(Empty())
        if self._config.ros_config.ros_launch_config.gazebo:
            self._reset_gazebo()
        self._clear_experience_values()
        while self.fsm_state != FsmState.Running \
                or self.observation is None \
                or self.terminal_state is None:
            self._run_shortly()
        self.observation = self._filter_observation(self.observation)
        self._current_experience = Experience(
            done=deepcopy(self.terminal_state),
            observation=deepcopy(self.observation),
            time_stamp=int(rospy.get_time() * 10 ** 3),
            info={}
        )
        self._previous_observation = deepcopy(self.observation)
        return self._current_experience, deepcopy(self.observation)

    def step(self, action: Action = None) -> Tuple[Experience, np.ndarray]:
        self._step += 1
        if action is not None:
            for index, msg in enumerate(adapt_action_to_twist(action)):
                self._action_publishers[index].publish(msg)
            if self._config.ros_config.action_topic == 'python':
                self._action = deepcopy(action)
        self._run_and_update_experience()
        return self._current_experience, deepcopy(self.observation)

    def remove(self) -> bool:
        return self._ros.terminate() == ProcessState.Terminated
