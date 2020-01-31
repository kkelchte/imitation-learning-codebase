#!/usr/bin/python3.7
import os
from typing import Union, List

import numpy as np
import rospy
from imitation_learning_ros_package.msg import RosSensor, RosAction
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import skimage.transform as sm
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray

from src.sim.common.actors import ActorConfig
from src.sim.common.data_types import Action, ActorType
from src.sim.ros.extra_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge

bridge = CvBridge()


def get_output_path() -> str:
    output_path = rospy.get_param('/output_path', '/tmp')
    if not output_path.startswith('/'):
        output_path = os.path.join(os.environ['HOME'], output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


def get_type_from_topic_and_actor_configs(actor_configs: List[ActorConfig], topic_name: str) -> ActorType:
    for actor_config in actor_configs:
        if actor_config.specs['command_topic'] == topic_name:
            return actor_config.type
    return ActorType.Unknown


def get_distance(a: Union[tuple, list], b: Union[tuple, list]) -> float:
    return np.sqrt(sum((np.asarray(a) - np.asarray(b)) ** 2))


def adapt_vector_to_odometry(data: Union[np.ndarray, list, tuple]) -> Odometry:
    """data is 1D numpy array in format [x, y, z] or [x, y, z, yaw] or [x, y, z, qx, qy, qz, qw]"""
    odometry = Odometry()
    odometry.pose.pose.position.x = data[0]
    odometry.pose.pose.position.y = data[1]
    odometry.pose.pose.position.z = data[2]
    if len(data) == 3:
        return odometry
    if len(data) == 4:
        qx, qy, qz, qw = quaternion_from_euler((0, 0, data[3]))
    else:
        qx, qy, qz, qw = data[3:]
    odometry.pose.pose.orientation.x = qx
    odometry.pose.pose.orientation.y = qy
    odometry.pose.pose.orientation.z = qz
    odometry.pose.pose.orientation.w = qw
    return odometry


def adapt_sensor_to_ros_message(data: np.ndarray, sensor_name: str) -> RosSensor:

    message = RosSensor()
    if 'waypoint' in sensor_name:
        message.waypoint = Float32MultiArray()
        message.waypoint.data = data.tolist()
    elif 'odom' in sensor_name:
        message.odometry = adapt_vector_to_odometry(data)
    return message


def adapt_twist_to_action(msg: Twist) -> Action:
    return Action(
        value=np.asarray(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z
            ]
        )
    )


def adapt_action_to_ros_message(action: Action) -> RosAction:
    msg = RosAction()
    msg.value = adapt_action_to_twist(action)
    msg.name = action.actor_name
    msg.type = action.actor_type
    return msg


def adapt_action_to_twist(action: Action) -> Union[Twist, None]:
    if action.value is None:
        return None
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z = \
        tuple(action.value)
    return twist


def resize_image(img: np.ndarray, sensor_stats: dict) -> np.ndarray:
    size = [
        sensor_stats['height'],
        sensor_stats['width'],
        sensor_stats['depth'],
    ]
    scale = [int(img.shape[i] / size[i]) for i in range(len(img.shape))]
    img = img[
          ::scale[0],
          ::scale[1],
          ::scale[2]
          ]
    return sm.resize(img, size, mode='constant').astype(np.float32)


def process_odometry(msg: Odometry, _=None) -> np.ndarray:
    return np.asarray([msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       msg.pose.pose.position.z,
                       msg.pose.pose.orientation.x,
                       msg.pose.pose.orientation.y,
                       msg.pose.pose.orientation.z,
                       msg.pose.pose.orientation.w])


def process_image(msg, sensor_stats: dict = None) -> np.ndarray:
    if sensor_stats['depth'] == 1:
        img = bridge.imgmsg_to_cv2(msg, 'passthrough')
        max_depth = float(sensor_stats['max_depth']) if 'max_depth' in sensor_stats.keys() else 4
        min_depth = float(sensor_stats['min_depth']) if 'min_depth' in sensor_stats.keys() else 0.1
        img = np.clip(img, min_depth, max_depth).astype(np.float32)
        # TODO add image resize and smoothing option
        print('WARNING: utils.py: depth image is not resized.')
        return img
    else:
        img = bridge.imgmsg_to_cv2(msg, 'rgb8')
        # TODO make automatic scale optional
        return resize_image(img, sensor_stats)


def process_compressed_image(msg, sensor_stats: dict = None) -> np.ndarray:
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    return resize_image(img, sensor_stats)


def process_laser_scan(msg, sensor_stats: dict = None) -> np.ndarray:
    field_of_view = int(sensor_stats['field_of_view']) if 'field_of_view' in sensor_stats.keys() else 90
    num_smooth_bins = int(sensor_stats['num_smooth_bins']) if 'num_smooth_bins' in sensor_stats.keys() else 4
    max_depth = float(sensor_stats['max_depth']) if 'max_depth' in sensor_stats.keys() else 4
    min_depth = float(sensor_stats['min_depth']) if 'min_depth' in sensor_stats.keys() else 0.1

    # clip at max and set too low values to nan
    ranges = [np.nan if r < min_depth else r for r in msg.ranges]
    ranges = [max_depth if r > max_depth else r for r in ranges]

    # clip left field-of-view degree range from 0:45 reversed with right 45degree range from the last 45:
    ranges = list(reversed(ranges[:int(field_of_view / 2)])) + list(
        reversed(ranges[-int(field_of_view / 2):]))
    # add some smoothing by averaging over 4 neighboring bins
    if num_smooth_bins != 1:
        ranges = [sum(ranges[i * num_smooth_bins: (i + 1) * num_smooth_bins]) / num_smooth_bins
                  for i in range(int(len(ranges) / num_smooth_bins))]

    # make it a numpy array
    return np.asarray(ranges)


def euler_from_quaternion(quaternion: tuple) -> tuple:
    return tuple(R.from_quat(quaternion).as_euler('XYZ'))


def quaternion_from_euler(euler: tuple) -> tuple:
    return tuple(R.from_euler('XYZ', euler, degrees=False).as_quat())
