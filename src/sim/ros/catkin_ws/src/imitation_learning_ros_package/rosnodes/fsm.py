#!/usr/bin/python3.7
import os
import time
from enum import IntEnum

import numpy as np
import rospy

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from src.sim.ros.extra_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge
from src.core.logger import get_logger, cprint
from src.sim.common.data_types import TerminalType
from src.sim.ros.src.utils import process_image, process_laser_scan, get_output_path

bridge = CvBridge()

"""FSM is a node that structures one episode of an ROS experiment.

The structure of the FSM is defined by the rosparam fsm_mode
SingleRun = 0 # start -(a)> Running -(b)> terminate
TakeOffRun = 1 # start -(a)> TakeOff -(c)> Running -(b)> terminate
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
    TakeOff = 10
    DriveBack = 20

    @classmethod
    def members(cls):
        return list(cls.__members__.keys())


class FsmMode(IntEnum):
    SingleRun = 0
    TakeOffRun = 1
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
        self._logger = get_logger(os.path.basename(__file__), self._output_path)

        self.mode = rospy.get_param('/fsm/fsm_mode')
        self._state_pub = rospy.Publisher(rospy.get_param('/fsm/state_topic'), String, queue_size=10)
        self._terminal_outcome_pub = rospy.Publisher(rospy.get_param('/fsm/terminal_topic'), String, queue_size=10)
        # self._pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._subscribe()
        rospy.init_node('fsm', anonymous=True)

        self._run_number = 0
        self._reset()

    def _subscribe(self):
        """Subscribe to relevant topics depending on the mode"""
        rospy.Subscriber(rospy.get_param('/fsm/finish_topic', '/finish'), Empty, self._shutdown_run,
                         callback_args=TerminalType.Unknown)
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

    def _start(self):
        """Define correct initial state depending on mode"""
        if self.mode == FsmMode.SingleRun:
            self._running()
        if self.mode == FsmMode.TakeOffRun:
            self._takeoff()
        if self.mode == FsmMode.TakeOverRun or self.mode == FsmMode.TakeOverRunDriveBack:
            self._takeover()

    def _reset(self, msg: Empty = None):
        """Add entrance of idle state all field variables are reset
        """
        self._delay_evaluation = rospy.get_param('world/delay_evaluation')
        self._is_shuttingdown = False
        self._set_state(FsmState.Unknown)
        self._start_time = -1
        self._current_pos = np.zeros((3,))
        self.travelled_distance = 0
        self.success = None
        if rospy.has_param('world/starting_height') and self.mode == FsmMode.TakeOffRun:
            self.starting_height = rospy.get_param('world/starting_height')
        # Params to define success and failure
        self._max_duration = rospy.get_param('world/max_duration', -1)
        self._collision_depth = rospy.get_param('robot/collision_depth', 0)
        self._max_travelled_distance = rospy.get_param('world/max_travelled_distance', -1)
        self._max_distance_from_start = rospy.get_param('world/max_distance_from_start', -1)
        self._goal = rospy.get_param('world/goal', {})  # TODO: check if this works.
        if True:
            print('****FSM: Settings:****')
            for name, value in [('_max_duration', self._max_duration), ('_collision_depth', self._collision_depth),
                          ('_max_travelled_distance', self._max_travelled_distance), ('mode', self.mode),
                          ('_max_distance_from_start', self._max_distance_from_start), ('_goal', self._goal)]:
                print(f'{name}: {value}')
            print('********')
        self._start()

    def _set_state(self, state: FsmState) -> None:
        cprint(f'set state: {state.name}', self._logger)
        self._state = state
        self._state_pub.publish(self._state.name)

    def _running(self, msg: Empty = None):
        self._set_state(FsmState.Running)
        self._run_number += 1
        if self._start_time == -1:
            self._start_time = rospy.get_time()

    def _takeoff(self):
        self._set_state(FsmState.TakeOff)

    def _takeover(self, msg: Empty = None):
        self._set_state(FsmState.TakenOver)

    def _shutdown_run(self, msg: Empty = None, outcome: TerminalType = TerminalType.Unknown):
        # Invocation for next run
        self._is_shuttingdown = True
        # pause_physics_client(EmptyRequest())
        cprint(f'Terminated with {outcome.name}', self._logger)
        if self.mode == FsmMode.TakeOverRunDriveBack:
            self._set_state(FsmState.DriveBack)
        else:
            self._set_state(FsmState.Terminated)
        self._terminal_outcome_pub.publish(outcome.name)

    def _check_time(self) -> float:
        run_duration_s = rospy.get_time() - self._start_time if self._start_time != -1 else 0
        if self._start_time != -1 and run_duration_s > self._max_duration != -1 and not self._is_shuttingdown:
            cprint(f'duration: {run_duration_s} > {self._max_duration}', self._logger)
            self._shutdown_run(outcome=TerminalType.Success)
        return run_duration_s

    def _update_state(self) -> bool:
        duration_s = self._check_time()
        return self._state is FsmState.Running \
            and not self._is_shuttingdown \
            and duration_s > self._delay_evaluation

    def _check_depth(self, data: np.ndarray) -> None:

        if np.amin(data) < self._collision_depth and not self._is_shuttingdown:
            cprint(f'Depth value {np.amin(data)} < {self._collision_depth}', self._logger)
            self._shutdown_run(outcome=TerminalType.Failure)

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

    def _check_distance(self, max_val: float, val: float, swap: bool = False) -> bool:
        if not swap:
            return max_val != -1 and max_val > val and not self._is_shuttingdown
        else:
            return max_val != -1 and max_val < val and not self._is_shuttingdown

    def _check_position(self, msg: Odometry) -> None:
        previous_pos = np.copy(self._current_pos)
        self._current_pos = np.asarray([msg.pose.pose.position.x,
                                        msg.pose.pose.position.y,
                                        msg.pose.pose.position.z])
        if self.mode == FsmMode.TakeOffRun and self._current_pos[2] >= self.starting_height - 0.1:
            self._running()

        if not self._update_state():
            return
        self.travelled_distance += np.sqrt(np.sum((previous_pos - self._current_pos) ** 2))
        self.distance_from_start = np.sqrt(np.sum(self._current_pos ** 2))

        if self._check_distance(self._max_travelled_distance, self.travelled_distance):
            cprint(f'Travelled distance {self.travelled_distance} > max {self._max_travelled_distance}',
                   self._logger)
            self._shutdown_run(outcome=TerminalType.Success)

        if self._check_distance(self._max_distance_from_start, self.distance_from_start):
            cprint(f'Max distance {self.distance_from_start} > max {self._max_distance_from_start}',
                   self._logger)
            self._shutdown_run(outcome=TerminalType.Success)

        if self._goal and not self._is_shuttingdown and \
                self._goal['x']['max'] > self._current_pos[0] > self._goal['x']['min'] and \
                self._goal['y']['max'] > self._current_pos[1] > self._goal['y']['min'] and \
                self._goal['z']['max'] > self._current_pos[2] > self._goal['z']['min']:
            cprint(f'Reached goal {self._goal} on location {self._current_pos}', self._logger)
            self._shutdown_run(outcome=TerminalType.Success)

    def _check_wrench(self, msg: WrenchStamped) -> None:
        if not self._update_state():
            return
        if msg.wrench.force.z < 1:
            cprint(f"found drag force: {msg.wrench.force.z}, so robot must be upside-down.", self._logger)
            self._shutdown_run(outcome=TerminalType.Failure)

    def run(self):
        rate = rospy.Rate(50)  # 10hz
        while not rospy.is_shutdown():
            self._state_pub.publish(self._state.name)
            rate.sleep()


if __name__ == "__main__":
    fsm = Fsm()
    fsm.run()
