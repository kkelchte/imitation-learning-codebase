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

from imitation_learning_ros_package.msg import RosReward, CombinedGlobalPoses
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

        self._rate_fps = rospy.get_param('/fsm/rate_fps', 60)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._run_number = 0
        self._reward_calculator = RewardCalculator()

        self._init_fields()  # define fields which require a resetting at each run
        self._state_pub = rospy.Publisher(rospy.get_param('/fsm/state_topic'), String, queue_size=10)
        self._rewards_pub = rospy.Publisher(rospy.get_param('/fsm/reward_topic'), RosReward, queue_size=10)
        self._subscribe()
        rospy.init_node('fsm', anonymous=True)
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
        if rospy.has_param('/robot/modified_state_topic'):
            rospy.Subscriber(rospy.get_param('/robot/modified_state_topic'),
                             eval(rospy.get_param('/robot/modified_state_type')),
                             self._set_modified_state)
        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            rospy.Subscriber(rospy.get_param('/fsm/go_topic', '/go'), Empty, self._running)
            rospy.Subscriber(rospy.get_param('/fsm/overtake_topic', '/overtake'), Empty, self._takeover)

        # used to detect whether motors are still running as they shutdown on flip over.
        if rospy.has_param('/robot/wrench_topic'):
            rospy.Subscriber(rospy.get_param('/robot/wrench_topic'), WrenchStamped, self._check_wrench)

    def _init_fields(self):
        self.mode = rospy.get_param('/fsm/fsm_mode')
        self._state = FsmState.Unknown
        self._occasion = 'unk'
        self._step_count = 0
        self._delay_evaluation = rospy.get_param('world/delay_evaluation')
        self._is_shuttingdown = False
        self._set_state(FsmState.Unknown)
        self._start_time = -1
        self._robot_info = {
            'current_position': None,
            'starting_position': None,
            'combined_global_poses': None
        }
        self.success = None
        # Params to define success and failure
        self._max_duration = rospy.get_param('/world/max_duration', -1)
        self._collision_depth = rospy.get_param('/robot/max_collision_depth', 0)
        self._goal = rospy.get_param('world/goal', {})
        self._reward_calculator.reset()

    def run(self):
        rate = rospy.Rate(self._rate_fps)
        while not rospy.is_shutdown():
            self._state_pub.publish(self._state.name)
            self._rewards_pub.publish(self._reward_calculator.get_reward(self._occasion,
                                                                         self._robot_info))
            if self._state == FsmState.Running:
                self._occasion = 'step'
            rate.sleep()
            self._step_count += 1
            if self._step_count % self._rate_fps == 0:
                msg = f"{rospy.get_time(): 0.0f}ms, state: {self._state.name}, shutting down:{self._is_shuttingdown}"
                cprint(msg, self._logger)

    ###############################
    # FSM state transition function
    ###############################
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
                    cprint(f'waiting for takeoff', self._logger)
                    client.wait_for_server()
                    client.send_goal(goal=TakeoffActionGoal())

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
        while rospy.get_time() - self._start_time < self._delay_evaluation:
            rospy.sleep(0.01)
        self._set_state(FsmState.Running)
        self._run_number += 1

    def _takeover(self, msg: Empty = None):
        self._set_state(FsmState.TakenOver)

    def _shutdown_run(self, msg: Empty = None):
        # Invocation for next run
        self._is_shuttingdown = True
        # pause_physics_client(EmptyRequest())
        if self.mode == FsmMode.TakeOverRunDriveBack:
            self._set_state(FsmState.DriveBack)
        else:
            self._set_state(FsmState.Terminated)

    ###############################
    # Callback functions
    ###############################
    def _set_modified_state(self, msg) -> None:
        if isinstance(msg, CombinedGlobalPoses):
            self._robot_info['combined_global_poses'] = msg
        else:
            raise NotImplemented(f'[FSM] failed to understand message of type {type(msg)}: \n {msg}')

    def _delay_evaluation(self) -> bool:
        """all reasons not to evaluate input at callback"""
        duration_s = rospy.get_time() - self._start_time if self._start_time != -1 else 0
        if self._is_shuttingdown or self._state != FsmState.Running:
            return True
        elif duration_s >= self._max_duration:
            self._occasion = 'out_of_time'
            self._shutdown_run()
            return True
        elif duration_s < self._delay_evaluation:
            return True
        else:
            return False

    def _check_depth(self, data: np.ndarray) -> None:
        if self._delay_evaluation():
            return
        if np.amin(data) < self._collision_depth:
            cprint(f'Depth value {np.amin(data)} < {self._collision_depth}', self._logger)
            self._occasion = 'on_collision'
            self._shutdown_run()

    def _check_depth_image(self, msg: Image) -> None:
        sensor_stats = {
            'depth': 1,
            'min_depth': 0.1,
            'max_depth': 5
        }
        processed_image = process_image(msg, sensor_stats)
        self._check_depth(processed_image)

    def _check_laser_scan(self, msg: LaserScan) -> None:
        sensor_stats = {
            'min_depth': 0.1,
            'max_depth': 5,
            'num_smooth_bins': 4,
            'field_of_view': 360
        }
        scan = process_laser_scan(msg, sensor_stats)
        self._check_depth(scan)

    def _check_wrench(self, msg: WrenchStamped) -> None:
        if self._delay_evaluation():
            return
        if msg.wrench.force.z < 0:
            cprint(f"found drag force: {msg.wrench.force.z}, so robot must be upside-down.", self._logger)
            self._occasion = 'on_collision'
            self._shutdown_run()

    def _check_position(self, msg: Odometry) -> None:
        if not self._delay_evaluation():
            return
        current_pos = np.asarray([msg.pose.pose.position.x,
                                  msg.pose.pose.position.y,
                                  msg.pose.pose.position.z])

        if self._goal is not None and \
                self._goal['x']['max'] > current_pos[0] > self._goal['x']['min'] and \
                self._goal['y']['max'] > current_pos[1] > self._goal['y']['min'] and \
                self._goal['z']['max'] > current_pos[2] > self._goal['z']['min']:
            cprint(f'Reached goal on location {current_pos}', self._logger)
            self._occasion = 'goal_reached'
            self._shutdown_run()


class RewardCalculator:
    """
    Create reward calculator which provides rewards depending on reward occasion according to reward mapping.
    Reward occasions: 'goal_reached', 'step', 'on_collision', 'out_of_time'
    Reward value types: 'iou' (requires CombinedGlobalPoses states), 'distance_from_start', 'distance_between_agents'
    Values are combined in weighted sum with weights specified by world.
    """
    def __init__(self):
        self._mapping = rospy.get_param('/world/reward', {})
        self.travelled_distance = 0
        self.distance_from_start = 0

    def get_reward(self, occasion: str, kwargs: dict) -> RosReward:
        # self.travelled_distance += np.sqrt(np.sum((previous_pos - self._current_pos) ** 2))
        # self.distance_from_start = np.sqrt(np.sum(self._current_pos ** 2))

        reward_mapping = self._mapping[occasion]
        termination_type = reward_mapping['termination']
        reward = 0
        for reward_type, reward_weight in reward_mapping['weights'].items():
            value = eval(f'get_{reward_type}(kwargs)') if reward_type != 'constant' else 1
            reward += reward_weight * value
        return RosReward(
            reward=reward,
            termination=termination_type
        )

    def reset(self):
        self.travelled_distance = 0
        self.distance_from_start = 0


if __name__ == "__main__":
    Fsm()
