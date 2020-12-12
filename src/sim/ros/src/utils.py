#!/usr/bin/python3.8
import os
from math import sqrt, cos, sin
from typing import Union, List, Tuple, Iterable, Sequence

import numpy as np
import rospy
from cv2 import cv2
from imitation_learning_ros_package.msg import CombinedGlobalPoses
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import skimage.transform as sm
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import Float32MultiArray

from src.core.data_types import Action
from cv_bridge import CvBridge

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
        output_path = os.path.join(os.environ['CODEDIR'], output_path)
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


def adapt_action_to_twist(action: Action) -> List[Twist]:
    values = action.value.flatten()
    if action.actor_name == 'tracking_fleeing_agent':
        twist_tracking = Twist()
        twist_fleeing = Twist()
        twist_tracking.linear.x, twist_tracking.linear.y, twist_tracking.linear.z = tuple(values[:3])
        twist_fleeing.linear.x, twist_fleeing.linear.y, twist_fleeing.linear.z = tuple(values[3:6])
        twist_tracking.angular.z = values[6]
        twist_fleeing.angular.z = values[7]
        return [twist_tracking, twist_fleeing]
    else:
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z = \
            tuple(values)
        return [twist]


def resize_image(img: np.ndarray, sensor_stats: dict) -> np.ndarray:
    if 'height' in sensor_stats.keys() and 'width' in sensor_stats.keys():
        size = [sensor_stats['height'], sensor_stats['width'], 3]
    else:
        return img
    if 'depth' in sensor_stats.keys():
        size[2] = sensor_stats['depth']
    scale = [max(int(img.shape[i] / size[i]), 1) for i in range(2)]
    img = img[::scale[0],
              ::scale[1],
              :]
    img = sm.resize(img, size, mode='constant').astype(np.float32)
    if size[-1] == 1:
        img = img.mean(axis=-1, keepdims=True)
    return img


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


def process_pose_stamped(msg: PoseStamped, _=None) -> np.ndarray:
    return np.asarray([msg.pose.position.x,
                       msg.pose.position.y,
                       msg.pose.position.z,
                       msg.pose.orientation.x,
                       msg.pose.orientation.y,
                       msg.pose.orientation.z,
                       msg.pose.orientation.w])


def process_image(msg: Image, sensor_stats: dict = None) -> np.ndarray:
    # if sensor_stats['depth'] == 1:
    #     img = bridge.imgmsg_to_cv2(msg, 'passthrough')
    #     max_depth = float(sensor_stats['max_depth']) if 'max_depth' in sensor_stats.keys() else 4
    #     min_depth = float(sensor_stats['min_depth']) if 'min_depth' in sensor_stats.keys() else 0.1
    #     img = np.clip(img, min_depth, max_depth).astype(np.float32)
    #     # TODO add image resize and smoothing option
    #     print('WARNING: utils.py: depth image is not resized.')
    #     return img
    # else:
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)
    return resize_image(img, sensor_stats) if sensor_stats is not None else img


def process_compressed_image(msg, sensor_stats: dict = None) -> np.ndarray:
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #img = cv2.flip(img, 0)
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


def process_twist(msg, sensor_stats: dict = None) -> Action:
    return Action(
        actor_name=sensor_stats['name'] if sensor_stats is not None and 'name' in sensor_stats.keys() else '',
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


def process_float32multi_array(msg: Float32MultiArray, stats: dict = None) -> np.ndarray:
    # TODO get layout out of msg
    return np.asarray(msg.data)


def process_combined_global_poses(msg: CombinedGlobalPoses, sensor_stats: dict = None) -> np.ndarray:
    return np.asarray([msg.tracking_x,
                       msg.tracking_y,
                       msg.tracking_z,
                       msg.fleeing_x,
                       msg.fleeing_y,
                       msg.fleeing_z,
                       msg.tracking_roll,
                       msg.tracking_pitch,
                       msg.tracking_yaw])


def array_to_combined_global_pose(data: Sequence) -> CombinedGlobalPoses:
    msg = CombinedGlobalPoses()
    msg.tracking_x = data[0]
    msg.tracking_y = data[1]
    msg.tracking_z = data[2]
    msg.fleeing_x = data[3]
    msg.fleeing_y = data[4]
    msg.fleeing_z = data[5]
    msg.tracking_roll = data[6]
    msg.tracking_pitch = data[7]
    msg.tracking_yaw = data[8]
    return msg


def euler_from_quaternion(quaternion: tuple) -> tuple:
    return tuple(R.from_quat(quaternion).as_euler('XYZ'))


def quaternion_from_euler(euler: tuple) -> tuple:
    return tuple(R.from_euler('XYZ', euler, degrees=False).as_quat())


def rotation_from_quaternion(quaternion: tuple) -> np.ndarray:
    return np.asarray(R.from_quat(quaternion).as_matrix())


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


def calculate_bounding_box(state: Sequence,
                           resolution: tuple = (100, 100),
                           focal_length: int = 30,
                           kx: int = 50,
                           ky: int = 50,
                           skew: float = 0) -> Tuple[Tuple[int, int], int, int]:
    """
    Fleeing agent in the frame of tracking agent is represented as a bounding box
    according to the relative pose of the fleeing with respect to the tracking agent.
    state: iterable with 9 values see CombinedGlobalPoses [position_tracking, position_fleeing, orientation_tracking]
    resolution: the size of the frame of the tracking agent
    focal_length, kx, ky, skew: intrinsic camera parameters of constructed frame of tracking agent.
    returns: (pos, w, h) pixel coordinates, height and width ounding box of fleeing agent.
    """
    x0 = resolution[0]//2
    y0 = resolution[1]//2
    agent0 = np.asarray(state[:3])
    agent1 = np.asarray(state[3:6])
    yaw = state[6]
    pitch = state[7]
    roll = state[8]
    rel_pos = get_relative_coordinates(agent0, agent1, yaw, pitch, roll)
    X = rel_pos[0]
    Y = rel_pos[1]
    Z = rel_pos[2]

    u = focal_length*X/Z
    v = focal_length*Y/Z

    # width and height of drone in meters
    w_drone = 0.2
    h_drone = 0.2

    pos = (u*kx+v*skew+x0, v*ky+y0)

    mx = Z/sqrt(Z**2 + X**2)
    my = Z/sqrt(Z**2 + Y**2)
    w = int(mx*kx*w_drone*focal_length/Z)
    h = int(my*ky*h_drone*focal_length/Z)
    return pos, w, h


def get_relative_coordinates(pos_agent0: np.ndarray,
                             pos_agent1: np.ndarray,
                             yaw: float,
                             pitch: float,
                             roll: float) -> np.ndarray:
    """
    pos_agent0: 3d global coordinates of agent 0 (tracking)
    pos_agent1: 3d global coordinates of agent 1 (fleeing)
    yaw: float global yaw turn pos_agent0
    roll: float global roll turn pos_agent0
    pitch: float global pitch turn pos_agent0
    return relative pose (rotation and translation) of agent1 (fleeing) in the frame of agent0 (tracking)
    """
    rot = np.array([[cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll),
                     cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],
                    [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll),
                     sin(yaw)*sin(pitch)*sin(roll)-cos(yaw)*sin(roll)],
                    [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]])
    relative_pos = np.squeeze(np.transpose(rot.dot(np.transpose(np.array(pos_agent1 - pos_agent0)))))
    return relative_pos


def distance(a: Sequence, b: Sequence) -> float:
    assert len(a) == len(b)
    return np.sqrt(sum((np.asarray(a).squeeze() - np.asarray(b).squeeze())**2)).item()


def calculate_iou_from_bounding_boxes(bounding_boxes) -> float:
    #  TODO calculate intersection over union from bounding boxes
    return 5

#########################################
# Helper functions for reward calculation
#########################################


def get_iou(info: dict) -> float:
    if info['combined_global_poses'] is None:
        return None
    state = [info['combined_global_poses'].tracking_x,
             info['combined_global_poses'].tracking_y,
             info['combined_global_poses'].tracking_z,
             info['combined_global_poses'].fleeing_x,
             info['combined_global_poses'].fleeing_y,
             info['combined_global_poses'].fleeing_z,
             info['combined_global_poses'].tracking_roll,
             info['combined_global_poses'].tracking_pitch,
             info['combined_global_poses'].tracking_yaw]
    try:
        bounding_boxes = calculate_bounding_box(state=np.asarray(state))
        result = calculate_iou_from_bounding_boxes(bounding_boxes)
    except:
        result = 5
    return result


def get_travelled_distance(info: dict) -> float:
    if info['previous_position'] is not None:
        increment = distance(info['current_position'],
                             info['previous_position'])
        info['travelled_distance'] = increment + info['travelled_distance'] \
            if info['travelled_distance'] is not None else increment
    info['previous_position'] = None  # make sure difference between the two is not calculated twice
    return info['travelled_distance']


def get_distance_from_start(info: dict) -> float:
    return distance(info['current_position'],
                    info['start_position'])


def get_distance_between_agents(info: dict) -> float:
    msg = info['combined_global_poses']
    return distance([msg.tracking_x, msg.tracking_y, msg.tracking_z],
                    [msg.fleeing_x, msg.fleeing_y, msg.fleeing_z]) if msg is not None else None
