#!/usr/bin/python3.8
import os
import time
from enum import IntEnum

import actionlib
import numpy as np
import rospy

from geometry_msgs.msg import WrenchStamped
from hector_uav_msgs.msg import TakeoffAction, TakeoffActionGoal
from std_msgs.msg import Empty, Float32
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from imitation_learning_ros_package.msg import RosReward
from cv_bridge import CvBridge
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension
from src.core.data_types import TerminationType
from src.sim.ros.src.utils import process_image, process_laser_scan, get_output_path

bridge = CvBridge()

"""FSM is a node that structures one episode of an ROS experiment.

The structure of the FSM is defined by the rosparam fsm_mode
SingleRun = 0 # start -(a)> Running -(b)> terminate
[DEPRECATED] TakeOffRun = 1 # start -(a)> TakeOff -(c)> Running -(b)> terminate
TakeOverRun = 2 # start -(a)> TakeOver (-(f)> terminate) <(e)-(d)> Running  
TakeOverRunDriveBack = 3 # start -(a)> TakeOver (-(f)> terminate) <(e)-(d)> Running (-(b)> terminate) \
    <(g)-(b)> DriveBack (-(e)> TakeOver)

Transitions:
(a) (internal) instant transition on startup (end of fsm.init)
(b) (internal) on automatically detected terminal state: collision, max_distance, max_duration, goal_state, ...
(c) (internal) on reaching desired height (-> Running state)
(d) (external) on USER's /GO signal (-> Running state)
(e) (external) on USER's /OVERTAKE signal (-> TakenOver state)
(f) (external) on USER's /FINISH signal (-> terminate)
(g) (external) on drive-back-services /GO signal (-> Running state)

Extension:
add start & stop code for each node so it does not need to run all the time.
"""


class FsmState(IntEnum):
    Unknown = -1
    Running = 0
    TakenOver = 1
    Terminated = 2
    DriveBack = 20

    @classmethod
    def members(cls):
        return list(cls.__members__.keys())


class FsmMode(IntEnum):
    SingleRun = 0
    TakeOverRun = 2
    TakeOverRunDriveBack = 3


class Fsm:

    def __init__(self):
        # wait for ROS to be ready
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/fsm/fsm_mode') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        self.mode = rospy.get_param('/fsm/fsm_mode')
        self._state = FsmState.Unknown
        self._init_fields()
        self._state_pub = rospy.Publisher(rospy.get_param('/fsm/state_topic'), String, queue_size=10)
        self._rewards_pub = rospy.Publisher(rospy.get_param('/fsm/reward_topic'), RosReward, queue_size=10)
        self._subscribe()
        rospy.init_node('fsm', anonymous=True)
        self._rate_fps = rospy.get_param('/fsm/rate_fps', 60)
        self.count = 0
        self._run_number = 0
        self.run()

    def _subscribe(self):
        """Subscribe to relevant topics depending on the mode"""
        rospy.Subscriber(rospy.get_param('/fsm/finish_topic', '/finish'), Empty, self._shutdown_run,
                         callback_args=TerminationType.Unknown)
        rospy.Subscriber(rospy.get_param('/fsm/reset_topic', '/reset'), Empty, self._reset)
        if rospy.has_param('/robot/depth_scan_topic'):
            rospy.Subscriber(rospy.get_param('/robot/depth_scan_topic'),
                             eval(rospy.get_param('/robot/depth_scan_type')),
                             self._check_laser_scan)
        if rospy.has_param('/robot/depth_image_topic'):
            rospy.Subscriber(rospy.get_param('/robot/depth_image_topic'),
                             eval(rospy.get_param('/robot/depth_image_type')),
                             self._check_depth_image)
        if rospy.has_param('/robot/odometry_topic'):
            rospy.Subscriber(rospy.get_param('/robot/odometry_topic'),
                             eval(rospy.get_param('/robot/odometry_type')),
                             self._check_position)
        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            rospy.Subscriber(rospy.get_param('/fsm/go_topic', '/go'), Empty, self._running)
            rospy.Subscriber(rospy.get_param('/fsm/overtake_topic', '/overtake'), Empty, self._takeover)

        # used to detect whether motors are still running as they shutdown on flip over.
        if rospy.has_param('/robot/wrench_topic'):
            rospy.Subscriber(rospy.get_param('/robot/wrench_topic'), WrenchStamped, self._check_wrench)

    def _init_fields(self):
        self._delay_evaluation = rospy.get_param('world/delay_evaluation')
        self._is_shuttingdown = False
        self._set_state(FsmState.Unknown)
        self._start_time = -1
        self._current_pos = np.zeros((3,))
        self.travelled_distance = 0
        self.distance_from_start = 0
        self.success = None
        # Params to define success and failure
        self._max_duration = rospy.get_param('world/max_duration', -1)
        self._collision_depth = rospy.get_param('robot/collision_depth', 0)
        self._max_travelled_distance = rospy.get_param('world/max_travelled_distance', -1)
        self._max_distance_from_start = rospy.get_param('world/max_distance_from_start', -1)
        self._goal = rospy.get_param('world/goal', {})
        self._world_rewards = rospy.get_param('world/reward', {})
        self._current_reward = 0
        self._termination = TerminationType.Unknown

    def _reset(self, msg: Empty = None):
        """Add entrance of idle state all field variables are reset
        """
        self._init_fields()
        cprint(f'resetting', self._logger)
        self._start()

    def _start(self):
        """Define correct initial state depending on mode"""
        if self.mode == FsmMode.SingleRun:
            self._robot = rospy.get_param('/robot/robot_type', '')
            if 'quadrotor_sim' in self._robot:
                action_name = rospy.get_param('/robot/takeoff_action', '/action/takeoff')

                def call_takeoff_action(n):
                    client = actionlib.SimpleActionClient(n, TakeoffAction)
                    client.wait_for_server()
                    client.send_goal(goal=TakeoffActionGoal())
                    client.wait_for_result()
                    cprint(f'takeoff: {client.get_result()}', self._logger)

                if isinstance(action_name, list):
                    for name in action_name:
                        call_takeoff_action(name)
                else:
                    call_takeoff_action(action_name)
                # rospy.wait_for_service('/enable_motors')
                # enable_motors_service = rospy.ServiceProxy('/enable_motors', EnableMotors)
                # enable_motors_service.call(True)

            self._running()
        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            self._takeover()

    def _set_state(self, state: FsmState) -> None:
        cprint(f'set state: {state.name}', self._logger)
        self._state = state

    def _running(self, msg: Empty = None):
        if self._start_time == -1:
            while rospy.get_time() == 0:
                time.sleep(0.1)
            self._start_time = rospy.get_time()
        while self._check_time() < self._delay_evaluation:
            rospy.sleep(0.01)
        self._set_state(FsmState.Running)
        self._termination = TerminationType.NotDone
        self._run_number += 1

    def _takeover(self, msg: Empty = None):
        self._set_state(FsmState.TakenOver)
        self._termination = TerminationType.Unknown

    def _shutdown_run(self, msg: Empty = None,
                      outcome: TerminationType = None, reward: float = None):
        # Invocation for next run
        self._is_shuttingdown = True
        self._termination = outcome if outcome is not None else TerminationType.Unknown
        self._current_reward = reward
        # pause_physics_client(EmptyRequest())
        if self.mode == FsmMode.TakeOverRunDriveBack:
            self._set_state(FsmState.DriveBack)
        else:
            self._set_state(FsmState.Terminated)

    def _check_time(self) -> float:
        run_duration_s = rospy.get_time() - self._start_time if self._start_time != -1 else 0
        if self._start_time != -1 and run_duration_s > self._max_duration != -1 and not self._is_shuttingdown:
            cprint(f'duration: {run_duration_s} > {self._max_duration}', self._logger)
            self._shutdown_run(outcome=TerminationType.Unknown,
                               reward=self._get_reward('duration'))
        return run_duration_s

    def _update_state(self) -> bool:
        duration_s = self._check_time()
        return self._state == FsmState.Running \
            and not self._is_shuttingdown \
            and duration_s > self._delay_evaluation

    def _check_depth(self, data: np.ndarray) -> None:
        if not self._update_state():
            return
        if np.amin(data) < self._collision_depth and not self._is_shuttingdown:
            cprint(f'Depth value {np.amin(data)} < {self._collision_depth}', self._logger)
            self._shutdown_run(outcome=TerminationType.Failure,
                               reward=self._get_reward('collision'))

    def _check_depth_image(self, msg: Image) -> None:
        if not self._update_state():
            return
        sensor_stats = {
            'depth': 1,
            'min_depth': 0.1,
            'max_depth': 5
        }
        processed_image = process_image(msg, sensor_stats)
        self._check_depth(processed_image)

    def _check_laser_scan(self, msg: LaserScan) -> None:
        if not self._update_state():
            return
        sensor_stats = {
            'min_depth': 0.1,
            'max_depth': 5,
            'num_smooth_bins': 4,
            'field_of_view': 360
        }
        scan = process_laser_scan(msg, sensor_stats)
        self._check_depth(scan)

    def _check_distance(self, max_val: float, val: float) -> bool:
        return max_val != -1 and max_val < val and not self._is_shuttingdown

    def _check_position(self, msg: Odometry) -> None:
        previous_pos = np.copy(self._current_pos)
        self._current_pos = np.asarray([msg.pose.pose.position.x,
                                        msg.pose.pose.position.y,
                                        msg.pose.pose.position.z])
        if not self._update_state():
            return
        self.travelled_distance += np.sqrt(np.sum((previous_pos - self._current_pos) ** 2))
        self.distance_from_start = np.sqrt(np.sum(self._current_pos ** 2))

        if self._check_distance(self._max_travelled_distance, self.travelled_distance):
            cprint(f'Travelled distance {self.travelled_distance} > max {self._max_travelled_distance}',
                   self._logger)
            self._shutdown_run(outcome=TerminationType.Failure,
                               reward=self._get_reward('distance'))

        if self._check_distance(self._max_distance_from_start, self.distance_from_start):
            cprint(f'Max distance {self.distance_from_start} > max {self._max_distance_from_start}',
                   self._logger)
            self._shutdown_run(outcome=TerminationType.Failure,
                               reward=self._get_reward('distance'))

        if self._goal and not self._is_shuttingdown and \
                self._goal['x']['max'] > self._current_pos[0] > self._goal['x']['min'] and \
                self._goal['y']['max'] > self._current_pos[1] > self._goal['y']['min'] and \
                self._goal['z']['max'] > self._current_pos[2] > self._goal['z']['min']:
            cprint(f'Reached goal on location {self._current_pos}', self._logger)
            self._shutdown_run(outcome=TerminationType.Success,
                               reward=self._get_reward('goal'))

    def _check_wrench(self, msg: WrenchStamped) -> None:
        if not self._update_state():
            return
        if msg.wrench.force.z < 0:
            cprint(f"found drag force: {msg.wrench.force.z}, so robot must be upside-down.", self._logger)
            self._shutdown_run(outcome=TerminationType.Failure,
                               reward=self._get_reward('collision'))

    def _get_reward(self, key: str) -> float:
        reward = None
        if key in self._world_rewards.keys():
            reward = self._world_rewards[key]
        if 'add_distance_from_start' in self._world_rewards.keys() and self._world_rewards['add_distance_from_start']:
            cprint("adding max distance to reward!")
            reward = self.distance_from_start if reward is None else reward + self.distance_from_start
        return reward

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._state_pub.publish(self._state.name)
            if 'step' in self._world_rewards.keys() and \
                    (self._state == FsmState.Running or self._state == FsmState.Terminated) and \
                    self._current_reward == 0:  # If reward is not defined yet, check for step reward
                self._current_reward = self._world_rewards['step']
            self._rewards_pub.publish(RosReward(
                termination=self._termination.name,
                reward=self._current_reward
            ))
            rate.sleep()
            self.count += 1
            if self.count % self._rate_fps == 0:
                msg = f"{rospy.get_time(): 0.0f}ms reward: {self._current_reward} shutting down:{self._is_shuttingdown}"
                cprint(msg, self._logger)


if __name__ == "__main__":
    Fsm()
