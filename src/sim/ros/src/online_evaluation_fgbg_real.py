import os
from sys import argv
import shutil
import time
import unittest
from copy import deepcopy
from typing import Union

import torch
from std_msgs.msg import Empty
from tqdm import tqdm
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import rospy
import subprocess
import shlex
import cv2 as cv
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Pose
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import fgbg

from src.core.utils import get_filename_without_extension, get_to_root_dir, get_date_time_tag, safe_wait_till_true
from src.core.data_types import TerminationType, SensorType, Action
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from src.sim.ros.src.utils import transform

WORLD = 'real_world'
TARGET = 'cone'  # gate line
NUMBER = 5
DS_TASK = 'velocities'  # 'velocities'  # waypoints
CHECKPOINT = os.path.join(os.environ['HOME'], 'code/contrastive-learning/data/best_down_stream', TARGET, DS_TASK)


if __name__ == '__main__':
    config = {
        'output_path': "real-bebop-fgbg",
        'robot_name': 'bebop_real',
        'gazebo': False,
        'fsm': True,
        'fsm_mode': 'TakeOverRun',
        'control_mapping': True,
        'control_mapping_config': 'mathias_controller_keyboard' if DS_TASK == "waypoints" else "keyboard_python",
        'april_tag_detector': False,
        'altitude_control': False,
        'mathias_controller_with_KF': True,
        'starting_height': 1.5,
        'keyboard': True,
        'mathias_controller_config_file_path_with_extension':
            f'{os.environ["CODEDIR"]}/src/sim/ros/config/actor/mathias_controller_with_KF_real_bebop.yml',
    }

    # spinoff roslaunch
    ros_process = RosWrapper(launch_file='load_ros.launch',
                             config=config,
                             visible=True)

    # Load model
    model = fgbg.DownstreamNet(output_size=(4,) if DS_TASK == 'velocities' else (3,),
                               encoder_ckpt_dir=CHECKPOINT)
    visualisation_topic = '/actor/mathias_controller/visualisation'
    subscribe_topics = [
        TopicConfig(topic_name=rospy.get_param('/robot/position_sensor/topic'),
                    msg_type=rospy.get_param('/robot/position_sensor/type')),
        TopicConfig(topic_name='/fsm/state',
                    msg_type='String'),
        TopicConfig(topic_name='/reference_pose', msg_type='PointStamped'),
        TopicConfig(topic_name=visualisation_topic,
                    msg_type='Image')
    ]
    publish_topics = [
        TopicConfig(topic_name='/fsm/reset', msg_type='Empty'),
    ]

    ros_topic = TestPublisherSubscriber(
        subscribe_topics=subscribe_topics,
        publish_topics=publish_topics
    )

    safe_wait_till_true('"/fsm/state" in kwargs["ros_topic"].topic_values.keys()',
                        True, 25, 0.1, ros_topic=ros_topic)

    index = 0
    while index < NUMBER:
        # create output json and hdf5 file
        json_data = {
            "velocities": [],
            "global_target_location": [],
            "relative_target_location": [],
            "global_drone_pose": [],
            "predictions": [],
            "rmse": []
        }
        hdf5_data = {"observation": []}

        print(f'start loop: {index} with resetting')
        # publish reset
        ros_topic.publishers['/fsm/reset'].publish(Empty())
        rospy.sleep(0.5)

        print(f'waiting in overtake state')
        while ros_topic.topic_values["/fsm/state"].data != FsmState.Running.name:
            rospy.sleep(0.5)

        waypoints = []
        print(f'waiting in running state')
        while ros_topic.topic_values["/fsm/state"].data != FsmState.TakenOver.name:
            if '/reference_pose' in ros_topic.topic_values.keys() \
                    and '/bebop/odom' in ros_topic.topic_values.keys():
                odom = ros_topic.topic_values[rospy.get_param('/robot/position_sensor/topic')]
                point = transform([ros_topic.topic_values['/reference_pose'].point],
                                  orientation=odom.pose.pose.orientation,
                                  translation=odom.pose.pose.position)[0]
                waypoints.append(point)
            rospy.sleep(0.5)
        if len(waypoints) != 0:
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([_.x for _ in waypoints],
                       [_.y for _ in waypoints],
                       [_.z for _ in waypoints], label='waypoints')
            ax.legend()
            plt.savefig(os.path.join(config['output_path'], f'image_{index}.jpg'))
        index += 1

    print('finished')
