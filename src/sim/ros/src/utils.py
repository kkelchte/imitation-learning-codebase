#!/usr/bin/python3.8
import os
from typing import Union, List, Tuple

import numpy as np
import rospy
from imitation_learning_ros_package.msg import RosSensor, RosAction
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import skimage.transform as sm
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from src.core.data_types import Action
from src.sim.ros.python3_ros_ws.src.vision_opencv.cv_bridge.python.cv_bridge import CvBridge

bridge = CvBridge()


def apply_noise_to_twist(twist: Twist, noise: np.ndarray) -> Twist():
    twist.linear.x += noise[0]
    twist.linear.y += noise[1]
    twist.linear.z += noise[2]
    twist.angular.x += noise[3]
    twist.angular.y += noise[4]
    twist.angular.z += noise[5]
    return twist


def get_output_path() -> str:
    output_path = rospy.get_param('/output_path', '/tmp')
    if not output_path.startswith('/'):
        output_path = os.path.join(os.environ['HOME'], output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


def get_current_actor() -> str:
    actor_command_topic = rospy.get_param('/control_mapping/mapping/Running/command')
    actor_name = actor_command_topic.split('cmd_vel')[0].split('/actor')[-1].split('/')[1]
    return actor_name


def get_distance(a: Union[tuple, list, np.ndarray], b: Union[tuple, list, np.ndarray]) -> float:
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


def adapt_action_to_ros_message(action: Action) -> RosAction:
    msg = RosAction()
    msg.value = adapt_action_to_twist(action)
    msg.name = action.actor_name
    return msg


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


def adapt_action_to_twist(action: Action) -> Union[Twist, None]:
    if action.value is None:
        return None
    values = action.value.flatten()
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z = \
        tuple(values)
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


def process_imu(msg: Imu, _=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear_acceleration = np.asarray([msg.linear_acceleration.x,
                                      msg.linear_acceleration.y,
                                      msg.linear_acceleration.z])
    orientation = np.asarray([msg.orientation.x,
                              msg.orientation.y,
                              msg.orientation.z])
    angular_velocity = np.asarray([msg.angular_velocity.x,
                                   msg.angular_velocity.y,
                                   msg.angular_velocity.z])
    return linear_acceleration, orientation, angular_velocity


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


def rotation_from_quaternion(quaternion: tuple) -> np.ndarray:
    return np.asarray(R.from_quat(quaternion).as_matrix())


def transform(points: List[np.ndarray],
              orientation: np.ndarray = np.eye(3),
              translation: np.ndarray = np.zeros((3,)),
              invert: bool = False) -> List[np.ndarray]:
    augmented = True
    lengths = [len(p) for p in points]
    assert min(lengths) == max(lengths)
    if points[0].shape[0] == 3:
        augmented = False
        points = [np.concatenate([p, np.ones(1,)]) for p in points]
    transformation = np.zeros((4, 4))
    transformation[0:3, 0:3] = orientation
    transformation[0:3, 3] = translation
    transformation[3, 3] = 1
    if invert:
        transformation = np.linalg.inv(transformation)
    return [np.matmul(transformation, p)[:3] if not augmented else np.matmul(transformation, p) for p in points]


def project(points: List[np.ndarray],
            fx: float = 1,
            fy: float = 1,
            cx: float = 1,
            cy: float = 1) -> List[np.ndarray]:
    lengths = [len(p) for p in points]
    assert min(lengths) == max(lengths)
    if points[0].shape[0] == 3:
        points = [p[:3] for p in points]
    intrinsic_camera_matrix = np.asarray([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])
    pixel_coordinates = [np.matmul(intrinsic_camera_matrix, p) for p in points]
    return [p / p[2] for p in pixel_coordinates]


