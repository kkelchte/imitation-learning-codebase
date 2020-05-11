#!/usr/bin/python3.7

""" Listen to the robot's odometry and create a figure of the trajectory.

Assuming background is taken from gazebo image-taker with following GUI camera settings:
<pose frame=''>0 0 50 0 1.57 1.57</pose>
<view_controller>ortho</view_controller>
<projection_type>orthographic</projection_type>

Resolution: 964x1630

Transformation with frames r: robot, c: camera, g: gazebo/global

Translation:
t^r_c = t^g_c + s^g_c . t^r_g
with t^g_c = [964/2 1630/2] ~ origin of global frame in camera frame
with t^r_g = odometry.pose.position.x, odometry.pose.position.y
with s^g_c scaling of global frame coordinates to camera frame

Rotation:
R^r_c = R^g_c * R^r_g
with R^r_g = Rotation matrix from odometry.pose.orientation.yaw
with R^g_c = [[0 -1 0][1 0 0][0 0 1]] OR [[0 1 0][-1 0 0][0 0 1]]
"""
import os
import sys
import time
from typing import Tuple

import numpy as np
from PIL import Image
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from src.core.logger import get_logger, cprint
from src.core.utils import camelcase_to_snake_format, get_date_time_tag
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.utils import get_output_path, euler_from_quaternion, get_distance, rotation_from_quaternion, \
    transform, project, get_current_actor


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
        self._world_name = rospy.get_param('/world/world_name')
        self._robot_type = rospy.get_param('/robot/robot_type', 'turtlebot_sim')
        self._gui_camera_height = rospy.get_param('/world/gui_camera_height',
                                                  20 if 'turtle' in self._robot_type else 50)
        self._background_file = rospy.get_param('/world/background_file', None)
        if self._background_file is None:
            cprint('Could not find background file so exit.', self._logger)
            sys.exit(0)
        if not self._background_file.startswith('/'):
            self._background_file = os.path.join(os.environ['HOME'], self._background_file)
        self._background_image = Image.open(self._background_file)

        # Camera intrinsics from assumption of horizontal and vertical field of view
        horizontal_field_of_view = 60 * 3.14 / 180
        vertical_field_of_view = 40 * 3.14 / 180
        width, height = self._background_image.size
        self._fx = -width/2*np.tan(horizontal_field_of_view/2)**(-1)
        self._fy = -height/2*np.tan(vertical_field_of_view/2)**(-1)
        self._cx = width/2
        self._cy = height/2

        # Camera extrinsics: orientation (assumption) and translation (extracted from filename)
        self._camera_global_orientation = np.asarray([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        if len(os.path.basename(self._background_file).split('_')) > 2:
            # extract camera position from background file name
            x, y, z = (float(x) for x in self._background_file.split('.')[0].split('_')[-3:])
            self._camera_global_translation = np.asarray([x, y, z])
        else:
            self._camera_global_translation = np.asarray([0, 0, 10])

        self._minimum_distance_px = 30
        self._local_frame = [np.asarray([0, 0, 0]),
                             np.asarray([0.17, 0, 0]),
                             np.asarray([0, 0.17, 0]),
                             np.asarray([0, 0, 0.17])]

        self._previous_position = None
        self._frame_points = []

        rospy.init_node('robot_mapper')
        self._rate = rospy.Rate(10)
        cprint(f'specifications: \n'
               f'cy: {self._cy}\n'
               f'cx: {self._cx}\n'
               f'fx: {self._fx}\n'
               f'fy: {self._fy}\n'
               f'camera_rotation: {self._camera_global_orientation}\n'
               f'camera_translation: {self._camera_global_translation}\n', self._logger)

    def _update_fsm_state(self, msg: String):
        self._fsm_state = FsmState[msg.data]
        if self._fsm_state == FsmState.Terminated:
            self._write_image()

    def _odometry_callback(self, msg: Odometry):
        robot_global_translation = np.asarray([msg.pose.pose.position.x,
                                               msg.pose.pose.position.y,
                                               msg.pose.pose.position.z])
        robot_global_orientation = rotation_from_quaternion((msg.pose.pose.orientation.x,
                                                             msg.pose.pose.orientation.y,
                                                             msg.pose.pose.orientation.z,
                                                             msg.pose.pose.orientation.w))
        #cprint(f'robot_global_translation: {robot_global_translation}', self._logger)
        #cprint(f'robot_global_orientation: {robot_global_orientation}', self._logger)
        points_global_frame = transform(points=self._local_frame,
                                        orientation=robot_global_orientation,
                                        translation=robot_global_translation)
        points_camera_frame = transform(points=points_global_frame,
                                        orientation=self._camera_global_orientation,
                                        translation=self._camera_global_translation,
                                        invert=True)  # camera_global transformation is given, but should be inverted
        points_image_frame = project(points=points_camera_frame,
                                     fx=self._fx,
                                     fy=self._fy,
                                     cx=self._cx,
                                     cy=self._cy)
        if self._previous_position is None:
            self._previous_position = points_image_frame[0]  # store origin of position
            self._frame_points.append(points_image_frame)
        elif get_distance(self._previous_position, points_image_frame[0]) > self._minimum_distance_px:
            self._previous_position = points_image_frame[0]  # store origin of position
            self._frame_points.append(points_image_frame)

    def _write_image(self):
        fig, ax = plt.subplots()
        ax.imshow(self._background_image)
        for _frame in self._frame_points:
            colors = ['red', 'green', 'blue']
            for index, p in enumerate(_frame[1:]):
                xmin = _frame[0][0]
                ymin = _frame[0][1]
                line = mlines.Line2D([xmin, p[0]], [ymin, p[1]], linewidth=1, color=colors[index])
                ax.add_line(line)
                # plt.scatter(p[0], p[1], s=2, color=colors[index])
        plt.axis('off')
        output_file = f'{self._output_path}/trajectories/{get_date_time_tag()}'
        try:
            actor_name = get_current_actor()
        except IndexError:
            pass
        except KeyError:
            pass
        else:
            if actor_name == 'dnn_actor':
                output_file += '_' + os.path.basename(
                    rospy.get_param('/actor/dnn_actor/specs/model_config/load_checkpoint_dir'))
            else:
                output_file += '_' + actor_name
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cprint(f'writing image to {output_file}', self._logger)
        fig.savefig(output_file+'.png')
        plt.clf()
        plt.close(fig)
        self._frame_points = []

    def run(self):
        cprint(f'started with rate {self._rate}', self._logger)
        while not rospy.is_shutdown():
            self._rate.sleep()


if __name__ == "__main__":
    robot_mapper = RobotMapper()
    robot_mapper.run()
