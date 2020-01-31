#!/usr/bin/python3.7

""" Listen to the robot's odometry and create a figure of the trajectory.

Assuming background is taken from gazebo image-taker with following GUI camera settings:
<pose frame=''>0 0 50 0 1.57 1.57</pose>
<view_controller>ortho</view_controller>
<projection_type>orthographic</projection_type>

Resolution: 964x1630

Transformation with frames r: robot, c: camera, g: gazebo/global

Translation:
t^r_c = t^g_c + t^r_g
with t^g_c = [964/2 1630/2]
with t^r_g = odometry.pose.position.x, odometry.pose.position.y

Rotation:
R^r_c = R^g_c * R^r_g
with R^r_g = Rotation matrix from odometry.pose.orientation.yaw
with R^g_c = [[0 -1 0][1 0 0][0 0 1]] OR [[0 1 0][-1 0 0][0 0 1]]
"""
import os
import sys
import time

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from src.core.logger import get_logger, cprint
from src.core.utils import camelcase_to_snake_format
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, euler_from_quaternion, get_distance


class RobotMapper:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        while not rospy.has_param('/output_path') and time.time() < start_time + max_duration:
            time.sleep(0.1)
        self._output_path = get_output_path()
        self._logger = get_logger(os.path.basename(__file__), self._output_path)

        # subscribe to fsm state
        if not rospy.has_param('/fsm/state_topic'):
            cprint(f'failed to find fsm topic so quit', self._logger)
            sys.exit(1)

        rospy.Subscriber(name=rospy.get_param('/fsm/state_topic'),
                         data_class=String,
                         callback=self._update_fsm_state)
        # subscribe to odometry or other pose estimation type (such as GPS, gt_pose, ...)
        odometry_type = rospy.get_param('/robot/odometry_type')
        callback = f'_{camelcase_to_snake_format(odometry_type)}_callback'
        assert callback in self.__dir__()
        rospy.Subscriber(name=rospy.get_param('/robot/odometry_topic'),
                         data_class=eval(odometry_type),
                         callback=eval(f'self.{callback}'))

        # fields
        self._fsm_state = FsmState.Unknown
        # default_transformation = [1, 0, 1, 0]
        # self._world_to_image_transformation = rospy.get_param('/world/transformation', default=default_transformation)
        self._min_step_distance = rospy.get_param('/world/min_step_distance', 0.7)
        self._previous_position = []
        self._initial_arrow = np.asarray([[0., 0.], [7., 0.], [7., 1.5], [9., 0.], [7., -1.5], [7., 0.]]) * 4
        self._positions = []
        self._optima = {  # keep track of smallest and largest value among positions (arrows)
            'min': {
                'x': np.nan,
                'y': np.nan
            },
            'max': {
                'x': np.nan,
                'y': np.nan
            }

        }
        self._rotate_global_to_local = np.asarray([[-1, 0], [0, 1]])
        self._background_file = 'src/sim/ros/gazebo/backgrounds/'+rospy.get_param('/world/world_name')+'.png' \
            if rospy.has_param('/world/world_name') \
            and os.path.isfile('src/sim/ros/gazebo/backgrounds/' + rospy.get_param('/world/world_name') + '.png') \
            else ''

        rospy.init_node('robot_mapper')
        self._rate = rospy.Rate(10)

    def _update_minima_maxima(self, arrow: np.ndarray) -> None:
        for optimum in ['min', 'max']:
            for axe_index, axe in enumerate(['x', 'y']):
                self._optima[optimum][axe] = eval(f'np.nan{optimum}(arrow[:, {axe_index}])')

    def _update_fsm_state(self, msg: String):
        self._fsm_state = FsmState[msg.data]
        if self._fsm_state == FsmState.Terminated:
            self._write_image()

    def _translate_position(self, x: float, y: float, z: float = 0) -> tuple:
        return ()

    def _rotate_orientation(self, odom: Odometry) -> np.array:
        _, _, yaw = euler_from_quaternion((odom.pose.pose.orientation.x,
                                           odom.pose.pose.orientation.y,
                                           odom.pose.pose.orientation.z,
                                           odom.pose.pose.orientation.w))
        robot_orientation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return np.matmul(self._rotate_global_to_local, robot_orientation)

    def _get_transformation_local_to_drone(self, odom: Odometry) -> np.ndarray:
        transformation = np.identity(3)
        transformation[0:2, 2] = self._translate_position(odom.pose.pose.position.x, odom.pose.pose.position.y)
        transformation[0:2, 0:2] = inv(self._rotate_orientation(odom))
        return transformation

    def _update_arrow(self, transformation: np.ndarray) -> None:
        homogenous_arrow = np.concatenate([np.transpose(self._initial_arrow),
                                           np.ones((1, self._initial_arrow.shape[0]))])
        transformed_arrow = np.matmul(transformation, homogenous_arrow)
        updated_arrow = np.transpose(transformed_arrow)[:, :2]
        self._update_minima_maxima(updated_arrow)
        self._positions.append(updated_arrow)

    def _odometry_callback(self, msg: Odometry):
        # adjust orientation towards current_waypoint
        # cprint(f'Received odometry: \n{msg}', self._logger)
        current_position = self._translate_position(msg.pose.pose.position.x, msg.pose.pose.position.y)
        if len(self._previous_position) == 0:
            self._previous_position = current_position[:]
        if get_distance(a=current_position, b=self._previous_position) < self._min_step_distance:
            return
        self._update_arrow(self._get_transformation_local_to_drone(msg))

    def _write_image(self):
        # figsize = (30, 30)
        # fig, ax = plt.subplots(1, figsize=figsize)
        # ax.imshow(np.ones((964, 1630, 3)))
        # ax.add_patch(patches.Polygon(arrow, linewidth=3, edgecolor=(1, 0, 0), facecolor='None'))
        
        figsize = (30, 30)
        fig, ax = plt.subplots(1, figsize=figsize)
        if self._background_file:
            image = plt.imread(self._background_file)
            ax.imshow(image)
        else:
            height = abs(self._optima['max']['x'] - self._optima['min']['x'])
            width = abs(self._optima['max']['y'] - self._optima['min']['y'])
            image = np.ones((int(height)+10, int(width)+10, 3)) \
                if not np.isnan(height) and not np.isnan(width) else np.ones((1000, 1000, 3))
            ax.imshow(image)

        for position in self._positions:
            ax.add_patch(patches.Polygon(position, linewidth=1, edgecolor=(1, 0, 0), facecolor='None'))
        plt.tight_layout()
        fig.savefig(os.path.join(self._output_path, 'trajectory.png'))

    def run(self):
        cprint(f'started with rate {self._rate}', self._logger)
        while not rospy.is_shutdown():
            self._rate.sleep()


if __name__ == "__main__":
    robot_mapper = RobotMapper()
    robot_mapper.run()
