#!/usr/bin/python3.7
from typing import Union

import numpy as np
import roslib
import skimage.transform as sm

from src.sim.common.data_types import Action

roslib.load_manifest('imitation_learning_ros_package')
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

bridge = CvBridge()


def adapt_twist_to_action(msg: Twist) -> Action:
    return Action(
        value=np.ndarray(
            (
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z
            )
        )
    )


def adapt_action_to_twist(action: Action) -> Twist:
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.linear.z, \
        twist.angular.x, twist.angular.y, twist.angular.z = action.value
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
    return sm.resize(img, size, mode='constant').astype(np.float16)


def process_image(msg, sensor_stats: dict = None) -> np.ndarray:
    num_channels = sensor_stats['depth']
    img = bridge.imgmsg_to_cv2(msg, 'rgb8' if num_channels == 3 else 'passthrough')
    return resize_image(img, sensor_stats)


def process_compressed_image(msg, sensor_stats: dict = None) -> np.ndarray:
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
    return resize_image(img, sensor_stats)


def process_laser_scan(msg, sensor_stats: dict = None) -> np.ndarray:
    field_of_view = sensor_stats['field_of_view'] if 'field_of_view' in sensor_stats.keys() else 90
    num_smooth_bins = sensor_stats['num_smooth_bins'] if 'num_smooth_bins' in sensor_stats.keys() else 4
    max_depth = sensor_stats['max_depth'] if 'max_depth' in sensor_stats.keys() else 4
    min_depth = sensor_stats['min_depth'] if 'min_depth' in sensor_stats.keys() else 0.1

    # clip at 4 and ignore 0's
    ranges = [max_depth if r > max_depth or r < min_depth else r for r in msg.ranges]

    # clip left field-of-view degree range from 0:45 reversed with right 45degree range from the last 45:
    ranges = list(reversed(ranges[:field_of_view / 2])) + list(
        reversed(ranges[-field_of_view / 2:]))
    # add some smoothing by averaging over 4 neighboring bins
    ranges = [sum(ranges[i * num_smooth_bins: (i + 1) * num_smooth_bins]) / num_smooth_bins
              for i in range(int(len(ranges) / num_smooth_bins))]

    # make it a numpy array
    return np.asarray(ranges).reshape((1, -1))
