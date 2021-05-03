#!/usr/bin/python3.8
import time
import os
from copy import deepcopy
from enum import IntEnum

import numpy as np
import rospy

from geometry_msgs.msg import WrenchStamped, PoseStamped, Point
from std_msgs.msg import Empty
from geometry_msgs.msg import WrenchStamped, PoseStamped, Pose, Point
from hector_uav_msgs.msg import TakeoffAction, TakeoffActionGoal
from std_msgs.msg import Empty, Float32
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from imitation_learning_ros_package.msg import RosReward, CombinedGlobalPoses
from cv_bridge import CvBridge
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension, camelcase_to_snake_format
from src.core.data_types import TerminationType, SensorType, FsmMode
from src.sim.ros.src.utils import process_image, process_laser_scan, get_output_path, get_travelled_distance, \
    get_distance_from_start, get_iou, get_distance_between_agents  # do not remove, used by reward.

bridge = CvBridge()

"""FSM is a node that structures one episode of an ROS experiment.

The structure of the FSM is defined by the rosparam /fsm/mode
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


class Fsm:

    def __init__(self):
        # wait for ROS to be ready
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/fsm/mode') and time.time() < stime + max_duration:
            time.sleep(0.1)

        self._rate_fps = rospy.get_param('/fsm/rate_fps', 60)
        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self._run_number = 0
        self._reward_calculator = RewardCalculator()

        self._init_fields()  # define fields which require a resetting at each run
        self._state_pub = rospy.Publisher('/fsm/state', String, queue_size=10)
        self._rewards_pub = rospy.Publisher('/fsm/reward', RosReward, queue_size=10)
        self._subscribe()
        rospy.init_node('fsm', anonymous=True)
        self.run()

    def _subscribe(self):
        """Subscribe to relevant topics depending on the mode"""
        rospy.Subscriber('/fsm/finish', Empty, self._shutdown_run,
                         callback_args=TerminationType.Unknown)
        rospy.Subscriber('/fsm/reset', Empty, self._reset)
        for sensor_type in [SensorType.position,
                            SensorType.collision,
                            SensorType.depth,
                            SensorType.modified_state]:
            if rospy.has_param(f'/robot/{sensor_type.name}_sensor'):
                msg_type = rospy.get_param(f'/robot/{sensor_type.name}_sensor/type')
                rospy.Subscriber(rospy.get_param(f'/robot/{sensor_type.name}_sensor/topic'),
                                 eval(msg_type),
                                 eval(f"self._check_{camelcase_to_snake_format(msg_type)}"))

        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            rospy.Subscriber('/fsm/go', Empty, self._running)
            rospy.Subscriber('/fsm/overtake', Empty, self._takeover)

    def _init_fields(self):
        self.mode = FsmMode[rospy.get_param('/fsm/mode')]
        self._state = FsmState.Unknown
        self._occasion = 'unk'
        self._step_count = 0
        self._delay_evaluation_time = rospy.get_param('/world/delay_evaluation', 1.)
        self._is_shuttingdown = False
        self._set_state(FsmState.Unknown)
        self._start_time = -1
        self._robot_info = {
            'current_position': None,
            'previous_position': None,
            'start_position': None,
            'combined_global_poses': None,
            'travelled_distance': None
        }
        self.success = None
        # Params to define success and failure
        self._max_duration = rospy.get_param('/world/max_duration', -1)
        self._collision_depth = rospy.get_param('/robot/min_collision_depth', 0.3)
        self._goal = rospy.get_param('/world/goal', None)
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
                msg = f"{rospy.get_time(): 0.0f}ms, state: {self._state.name}, occasion: {self._occasion}"
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
            self._running()
        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            self._takeover()

    def _set_state(self, state: FsmState) -> None:
        if state.name != self._state.name:
            cprint(f'set state: {state.name}', self._logger)
        self._state = state

    def _running(self, msg: Empty = None):
        if self._start_time == -1:
            while rospy.get_time() == 0:
                time.sleep(0.1)
            self._start_time = rospy.get_time()
        while rospy.get_time() - self._start_time < self._delay_evaluation_time:
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
    def _delay_evaluation(self) -> bool:
        """all reasons not to evaluate input at callback"""
        duration_s = rospy.get_time() - self._start_time if self._start_time != -1 else 0
        if self._is_shuttingdown or self._state != FsmState.Running:
            return True
        elif duration_s >= self._max_duration != -1:
            self._occasion = 'out_of_time'
            self._shutdown_run()
            return True
        elif duration_s < self._delay_evaluation_time:
            return True
        else:
            return False

    def _check_depth(self, data: np.ndarray) -> None:
        if self._delay_evaluation():
            return
        if np.amin(data) < self._collision_depth:
            self._occasion = 'on_collision'
            self._shutdown_run()

    def _check_position(self, position: Point) -> None:
        """
        position is point with three dimension: x, y, z
        """
        if self._delay_evaluation():
            return
        current_pos = np.asarray([position.x,
                                  position.y,
                                  position.z])
        if self._robot_info['current_position'] is not None:
            self._robot_info['previous_position'] = deepcopy(self._robot_info['current_position'])
        self._robot_info['current_position'] = deepcopy(current_pos)
        if self._robot_info['start_position'] is None:
            self._robot_info['start_position'] = deepcopy(current_pos)
        if self._goal is not None and \
                self._goal['x']['max'] > current_pos[0] > self._goal['x']['min'] and \
                self._goal['y']['max'] > current_pos[1] > self._goal['y']['min'] and \
                self._goal['z']['max'] > current_pos[2] > self._goal['z']['min']:
            cprint(f'Reached goal on location {current_pos}', self._logger)
            self._occasion = 'goal_reached'
            self._shutdown_run()

    def _check_image(self, msg: Image) -> None:
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

    def _check_wrench_stamped(self, msg: WrenchStamped) -> None:
        if self._delay_evaluation():
            return
        if msg.wrench.force.z < 0:
            cprint(f"found drag force: {msg.wrench.force.z}, so robot must be upside-down.", self._logger)
            self._occasion = 'on_collision'
            self._shutdown_run()

    def _check_pose_stamped(self, msg: PoseStamped) -> None:
        self._check_position(msg.pose.position)

    def _check_odometry(self, msg: Odometry) -> None:
        self._check_position(msg.pose.pose.position)

    def _check_combined_global_poses(self, msg: CombinedGlobalPoses) -> None:
        if self._delay_evaluation():
            return
        if isinstance(msg, CombinedGlobalPoses):
            self._robot_info['combined_global_poses'] = msg
        else:
            raise NotImplemented(f'[FSM] failed to understand message of type {type(msg)}: \n {msg}')


class RewardCalculator:
    default_reward = {
        'unk': {'weights': {'constant': 0},
                'termination': TerminationType.Unknown.name},
        'step': {'weights': {'constant': 0},
                 'termination': TerminationType.NotDone.name},
        'on_collision': {'weights': {'constant': 0},
                         'termination': TerminationType.Failure.name},
        'goal_reached': {'weights': {'constant': 0},
                         'termination': TerminationType.Success.name},
        'out_of_time': {'weights': {'constant': 0},
                        'termination': TerminationType.Done.name},
    }
    """
    Create reward calculator which provides rewards depending on reward occasion according to reward mapping.
    Reward occasions: 'goal_reached', 'step', 'on_collision', 'out_of_time'
    Reward value types: 'constant', 'iou' (requires CombinedGlobalPoses states),
    'distance_from_start', 'distance_between_agents'
    Values are combined in weighted sum with weights specified by world.
    """
    def __init__(self):
        self._logger = get_logger('reward', get_output_path())
        self._mapping = rospy.get_param('/world/reward', self.default_reward)
        if 'unk' not in self._mapping.keys():  # add default unknown reward
            self._mapping['unk'] = {'weights': {'constant': 0},
                                    'termination': TerminationType.Unknown.name}
        cprint(f'starting with mapping: {self._mapping}', self._logger)
        self._count = 0

    def get_reward(self, occasion: str, kwargs: dict) -> RosReward:
        if occasion not in self._mapping.keys():
            occasion = 'unk'
        reward_mapping = self._mapping[occasion]
        termination_type = reward_mapping['termination']
        reward = 0
        for reward_type, reward_weight in reward_mapping['weights'].items():
            value = eval(f'get_{reward_type}(kwargs)') if reward_type != 'constant' else 1
            if value is not None:
                reward += reward_weight * value
        self._count += 1
        if self._count % 60 == 0 or termination_type in ['Failure', 'Success']:
            cprint(f'occasion: {occasion}, '
                   f'reward: {reward}, '
                   f'termination: {termination_type}', self._logger, msg_type=MessageType.debug)
        return RosReward(
            reward=reward,
            termination=termination_type
        )

    def reset(self):
        self._count = 0
        cprint('resetting', self._logger)
        self._mapping = rospy.get_param('/world/reward', self.default_reward)
        if 'unk' not in self._mapping.keys():  # add default unknown reward
            self._mapping['unk'] = {'weights': {'constant': 0},
                                    'termination': TerminationType.Unknown.name}


if __name__ == "__main__":
    Fsm()
